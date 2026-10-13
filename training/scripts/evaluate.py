#!/usr/bin/env python3
"""Evaluate a model on the test split with the exact runtime prompt and generation defaults.

Backends:
  --backend hf      --model PATH_OR_ID [--adapter ADAPTER_DIR]   (transformers, GPU recommended)
  --backend mlx     --model MLX_MODEL_DIR [--adapter MLX_ADAPTER] (Apple Silicon, mlx-lm)
  --backend openai  --base-url http://localhost:8080/v1          (llama.cpp llama-server or any
                                                                  OpenAI-compatible server; this is how
                                                                  the Q4_K_M GGUF is evaluated)

Generation defaults follow shared/CONTRACTS.md §5: temperature 0.3, top_p 0.9, top_k 40,
repeat penalty 1.1, max 1200 new tokens for answers; temperature 0.0 and max 256 tokens for memory
extraction / summarisation records. Answers are citation-validated as on device and scored with
bible_ft.metrics (RAG metrics, plus JSON-validity / type-key match for memory extraction and
word-limit / format checks for summaries). Writes ``predictions.jsonl``, ``metrics.json`` and ``report.md``
into --out-dir.

Examples:
  python training/scripts/evaluate.py --backend hf --model google/gemma-4-E2B-it --label base
  python training/scripts/evaluate.py --backend hf --model google/gemma-4-E2B-it \\
      --adapter training/outputs/gemma4-e2b-bible-qlora/adapter --label finetuned
  python training/scripts/evaluate.py --backend mlx --model training/outputs/mlx/gemma-4-E2B-it-4bit \
      --adapter training/outputs/mlx/adapter --label mlx-finetuned
  llama-server -m bible-gemma-q4_k_m.gguf -c 6144 --port 8080 &
  python training/scripts/evaluate.py --backend openai --base-url http://localhost:8080/v1 --label q4_k_m
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from bible_ft import TRAINING_DIR
from bible_ft.citations import extract_citations, validate_citations
from bible_ft.jsonl import read_jsonl, write_jsonl
from bible_ft.metrics import (
    EvalItem,
    compute_memory_metrics,
    compute_metrics,
    compute_summary_metrics,
    format_tag,
    is_refusal,
)
from bible_ft.schema import is_rag, is_refusal_record, prompt_messages, task_of
from bible_ft.thinking import strip_thinking

GEN_DEFAULTS = {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "max_new_tokens": 1200}
AUX_GEN = {"temperature": 0.0, "max_new_tokens": 256}  # §5: memory extraction / summarisation

# generate(messages, deterministic, max_new_tokens) -> text
Generator = Callable[[list[dict[str, str]], bool, int], str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["hf", "mlx", "openai"])
    p.add_argument("--rescore", type=Path, metavar="EVAL_DIR",
                   help="recompute metrics/report from EVAL_DIR/predictions.jsonl without generating")
    p.add_argument("--test", type=Path, default=TRAINING_DIR / "outputs" / "splits" / "test.jsonl")
    p.add_argument("--label", default="model", help="name used in reports (base / finetuned / q4_k_m)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: training/outputs/eval/<label>")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", help="hf: model id or path; openai: model name sent to the server")
    p.add_argument("--adapter", help="hf: optional PEFT adapter directory; mlx: optional mlx-lm adapter directory")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key (optional)")
    p.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=GEN_DEFAULTS["top_p"])
    p.add_argument("--top-k", type=int, default=GEN_DEFAULTS["top_k"])
    p.add_argument("--repeat-penalty", type=float, default=GEN_DEFAULTS["repeat_penalty"])
    p.add_argument("--max-new-tokens", type=int, default=GEN_DEFAULTS["max_new_tokens"])
    p.add_argument("--greedy", action="store_true", help="deterministic decoding (temperature 0) for A/B runs")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args(argv)


def hf_generator(args: argparse.Namespace) -> Generator:
    import torch
    from transformers import AutoProcessor

    from train_qlora import load_auto_model_class

    if not args.model:
        raise SystemExit("--model is required for --backend hf")
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.adapter or args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    model = load_auto_model_class().from_pretrained(args.model, dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(messages: list[dict[str, str]], deterministic: bool, max_new_tokens: int) -> str:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False,  # the on-device template renders with thinking off
        ).to(model.device)
        sampling = (
            {"do_sample": False}
            if args.greedy or deterministic
            else {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
        )
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, repetition_penalty=args.repeat_penalty, **sampling
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

    return generate


def mlx_generator(args: argparse.Namespace) -> Generator:
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    if not args.model:
        raise SystemExit("--model is required for --backend mlx")
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    penalties = make_logits_processors(repetition_penalty=args.repeat_penalty)

    def generate(messages: list[dict[str, str]], deterministic: bool, max_new_tokens: int) -> str:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        temp = 0.0 if args.greedy or deterministic else args.temperature
        sampler = make_sampler(temp=temp, top_p=args.top_p, top_k=args.top_k)
        text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, sampler=sampler,
                            logits_processors=penalties, verbose=False)
        return text.strip()

    return generate


def openai_generator(args: argparse.Namespace) -> Generator:
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get(args.api_key_env):
        headers["Authorization"] = f"Bearer {key}"

    def generate(messages: list[dict[str, str]], deterministic: bool, max_new_tokens: int) -> str:
        body = {
            "model": args.model or "local",
            "messages": messages,
            "temperature": 0.0 if args.greedy or deterministic else args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,  # llama-server extension
            "repeat_penalty": args.repeat_penalty,  # llama-server extension
            "max_tokens": max_new_tokens,
            "seed": args.seed,
            # llama-server (and vLLM) default Gemma 4 to thinking mode, which adds <|think|> to the system turn;
            # render exactly the prompt the model was trained on and the app sends.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)
        return (payload["choices"][0]["message"].get("content") or "").strip()

    return generate


def evaluate(records: list[dict[str, Any]], generate: Generator, max_new_tokens: int) -> list[dict[str, Any]]:
    predictions = []
    for i, record in enumerate(records, start=1):
        started = time.perf_counter()
        if not is_rag(record):
            raw = generate(prompt_messages(record), True, AUX_GEN["max_new_tokens"])
            output, thought = strip_thinking(raw)
            predictions.append({
                "id": record["id"], "task": task_of(record), "category": record["category"],
                "gold": record["answer"], "raw_output": raw, "output": output, "had_thinking": thought,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"[{i}/{len(records)}] {record['id']} ({task_of(record)})")
            continue
        n_sources = len(record["inputs"]["sources"])
        raw = generate(prompt_messages(record), False, max_new_tokens)
        elapsed = time.perf_counter() - started
        output, thought = strip_thinking(raw)  # score what the app would show
        result = validate_citations(output, n_sources)
        predictions.append({
            "id": record["id"],
            "task": "rag_answer",
            "category": record["category"],
            "expect_refusal": is_refusal_record(record),
            "gold_cited": extract_citations(record["answer"]),
            "format": format_tag(record["tags"]),
            "source_count": n_sources,
            "raw_output": raw,
            "output": output,
            "had_thinking": thought,
            "validated_output": result.text,
            "cited": result.cited,
            "removed": result.removed,
            "predicted_refusal": is_refusal(output, n_sources),
            "seconds": round(elapsed, 2),
        })
        print(f"[{i}/{len(records)}] {record['id']} cited={result.cited} removed={result.removed}")
    return predictions


def metrics_from_predictions(all_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [p for p in all_predictions if p.get("task", "rag_answer") == "rag_answer"]
    memory = [p for p in all_predictions if p.get("task") == "memory_extraction"]
    summaries = [p for p in all_predictions if p.get("task") == "summarization"]
    items = [
        EvalItem(
            prediction=p.get("output", p["raw_output"]),
            source_count=p["source_count"],
            expect_refusal=p["expect_refusal"],
            gold_cited=tuple(p["gold_cited"]),
            format=p["format"],
        )
        for p in predictions
    ]
    overall = compute_metrics(items).as_dict()
    overall["thinking_leak_rate"] = (
        sum(bool(p.get("had_thinking")) for p in all_predictions) / len(all_predictions) if all_predictions else 0.0
    )
    by_category: dict[str, Any] = {}
    for category in sorted({p["category"] for p in predictions}):
        subset = [it for it, p in zip(items, predictions) if p["category"] == category]
        by_category[category] = compute_metrics(subset).as_dict()
    return {
        "overall": overall,
        "by_category": by_category,
        "memory_extraction": compute_memory_metrics([p.get("output", p["raw_output"]) for p in memory], [p["gold"] for p in memory]),
        "summarization": compute_summary_metrics([p.get("output", p["raw_output"]) for p in summaries]),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(label: str, metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    lines = [f"# Evaluation: {label}", "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in metrics["overall"].items()]
    for task in ("memory_extraction", "summarization"):
        if metrics.get(task, {}).get("n"):
            lines += ["", f"## {task}", "", "| Metric | Value |", "|---|---:|"]
            lines += [f"| {k} | {fmt(v)} |" for k, v in metrics[task].items()]
    lines += ["", "## By category", "", "| Category | n | refusal ok | false refusal | coverage | validity |", "|---|---:|---:|---:|---:|---:|"]
    for cat, m in metrics["by_category"].items():
        lines.append(
            f"| {cat} | {m['n']} | {fmt(m['correct_refusal_rate'])} | {fmt(m['false_refusal_rate'])} "
            f"| {fmt(m['citation_coverage'])} | {fmt(m['citation_validity_rate'])} |"
        )
    failures = [
        p for p in predictions
        if p.get("task", "rag_answer") == "rag_answer"
        and (p["removed"] or p["predicted_refusal"] != p["expect_refusal"])
    ]
    lines += ["", f"## Failures ({len(failures)})", ""]
    for p in failures[:25]:
        why = []
        if p["removed"]:
            why.append(f"invalid citations {p['removed']}")
        if p["predicted_refusal"] != p["expect_refusal"]:
            why.append("missed refusal" if p["expect_refusal"] else "false refusal")
        lines += [f"### {p['id']} — {', '.join(why)}", "", "```", p["raw_output"][:1200], "```", ""]
    return "\n".join(lines) + "\n"


def rescore(out_dir: Path) -> int:
    """Re-apply the current refusal heuristic and metrics to saved predictions (e.g. after a metrics fix)."""
    predictions = list(read_jsonl(out_dir / "predictions.jsonl"))
    for p in predictions:
        if p.get("task", "rag_answer") == "rag_answer":
            p["predicted_refusal"] = is_refusal(p.get("output", p["raw_output"]), p["source_count"])
    previous = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_from_predictions(predictions)
    metrics["label"], metrics["generation"] = previous.get("label"), previous.get("generation")
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(metrics["label"], metrics, predictions), encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rescore:
        return rescore(args.rescore)
    if not args.backend:
        raise SystemExit("--backend is required unless --rescore is given")
    records = list(read_jsonl(args.test))[: args.limit]
    generate = {"hf": hf_generator, "mlx": mlx_generator, "openai": openai_generator}[args.backend](args)
    predictions = evaluate(records, generate, args.max_new_tokens)
    metrics = metrics_from_predictions(predictions)
    metrics["label"] = args.label
    metrics["generation"] = {
        "greedy": args.greedy, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "repeat_penalty": args.repeat_penalty, "max_new_tokens": args.max_new_tokens,
    }

    out_dir = args.out_dir or TRAINING_DIR / "outputs" / "eval" / args.label
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(args.label, metrics, predictions), encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
