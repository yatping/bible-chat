#!/usr/bin/env python3
"""Render the golden prompt cases through the real Gemma chat template.

For every case in shared/fixtures/prompt_golden.json: build messages with bible_ft.prompt_format
(asserting they equal ``expected_messages``), then render them with
``processor.apply_chat_template(..., add_generation_prompt=True, tokenize=False)``.

The output JSON lets the iOS side check that llama.cpp's GGUF-embedded template renders the
same string the model was trained on (the merged model / GGUF should embed the same template).

Example:
  python training/scripts/dump_chat_template.py --model google/gemma-4-E2B-it
  python training/scripts/dump_chat_template.py --model training/outputs/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from bible_ft import SHARED_DIR, TRAINING_DIR
from bible_ft.prompt_format import build_messages_from_inputs

GOLDEN = SHARED_DIR / "fixtures" / "prompt_golden.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-4-E2B-it", help="model id or local dir with the processor")
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--out", type=Path, default=TRAINING_DIR / "outputs" / "chat_template_render.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = json.loads(args.golden.read_text(encoding="utf-8"))["cases"]

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    rendered = []
    for case in cases:
        messages = build_messages_from_inputs(case["input"])
        if messages != case["expected_messages"]:
            raise SystemExit(f"prompt_format does not match golden case {case['name']!r}")
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        rendered.append({"name": case["name"], "rendered": text})

    template = getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None)
    out = {"model": args.model, "chat_template": template, "cases": rendered}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rendered)} rendered cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
