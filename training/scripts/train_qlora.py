#!/usr/bin/env python3
"""QLoRA fine-tune Gemma 4 E2B on the grounded-answering dataset (GPU required).

Follows Google's Gemma QLoRA recipe: 4-bit NF4 base (double quant), LoRA r=16/alpha=16,
``modules_to_save=["lm_head", "embed_tokens"]`` with weight tying, TRL ``SFTTrainer`` with a
custom collator. Loss is computed ONLY on the final assistant turn: system prompt, replayed
history (recent_messages) and the context block are masked with -100.

Every record is rendered with ``bible_ft.schema.to_messages`` -> the exact runtime prompt
(shared/CONTRACTS.md §1) -> ``processor.apply_chat_template``.

Example (single 24 GB GPU):
  python training/scripts/train_qlora.py \\
      --train training/outputs/splits/train.jsonl --val training/outputs/splits/val.jsonl \\
      --output-dir training/outputs/gemma4-e2b-bible-qlora
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from bible_ft import TRAINING_DIR
from bible_ft.jsonl import read_jsonl
from bible_ft.schema import to_messages

DEFAULT_MODEL = "google/gemma-4-E2B-it"
# Language-model projections only (skips the vision/audio towers). Use "all-linear" for the recipe default.
DEFAULT_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", type=Path, default=TRAINING_DIR / "outputs" / "splits" / "train.jsonl")
    p.add_argument("--val", type=Path, default=TRAINING_DIR / "outputs" / "splits" / "val.jsonl")
    p.add_argument("--output-dir", type=Path, default=TRAINING_DIR / "outputs" / "gemma4-e2b-bible-qlora")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", default="constant")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES,
                   help='regex or "all-linear" (recipe default; also adapts vision/audio towers)')
    p.add_argument("--no-modules-to-save", action="store_true",
                   help="do not fully train lm_head/embed_tokens (saves a lot of VRAM and adapter size)")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto",
                   help="compute dtype; auto = bf16 if supported else fp16")
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--dry-run", action="store_true", help="render + mask the first examples, then exit (no model)")
    return p.parse_args(argv)


def load_messages(path: Path) -> list[dict[str, Any]]:
    return [{"messages": to_messages(r)} for r in read_jsonl(path)]


class AssistantOnlyCollator:
    """Tokenise chat examples and mask every token except the final assistant answer.

    The prompt part is rendered with ``add_generation_prompt=True`` (system + history + context
    block + model-turn header); everything up to that length gets label -100. The answer and the
    template's end-of-turn marker are learned, so the model learns to stop.
    """

    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id

    def encode(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        assert messages[-1]["role"] == "assistant", "last message must be the gold answer"
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt_text = self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if full[: len(prompt)] != prompt:
            raise ValueError("chat template prompt is not a token prefix of the full conversation")
        labels = [-100] * len(prompt) + full[len(prompt):]
        if len(prompt) >= self.max_length:
            # Truncation keeps the start, which would drop the whole answer: the example teaches nothing.
            print(f"WARNING: prompt has {len(prompt)} tokens >= max_length {self.max_length}; answer truncated away")
        return full[: self.max_length], labels[: self.max_length]

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self.encode(ex["messages"]) for ex in examples]
        width = max(len(ids) for ids, _ in encoded)
        input_ids, attention, labels = [], [], []
        for ids, labs in encoded:
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_auto_model_class() -> Any:
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        if hasattr(transformers, name):
            return getattr(transformers, name)
    raise RuntimeError("no suitable AutoModel class in this transformers version")


def supported_kwargs(cls: Any, **kwargs: Any) -> dict[str, Any]:
    """Drop kwargs a (possibly older) library version doesn't accept, with a notice."""
    params = inspect.signature(cls).parameters
    kept = {k: v for k, v in kwargs.items() if k in params}
    for k in kwargs.keys() - kept.keys():
        print(f"NOTE: {cls.__name__} does not accept {k!r}; skipped")
    return kept


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = load_messages(args.train)
    val_rows = load_messages(args.val) if args.val.exists() else []
    print(f"train={len(train_rows)} val={len(val_rows)}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    processor = AutoProcessor.from_pretrained(args.model)
    collator = AssistantOnlyCollator(processor, args.max_length)

    if args.dry_run:
        for row in train_rows[:3]:
            ids, labels = collator.encode(row["messages"])
            learned = [t for t, l in zip(ids, labels) if l != -100]
            print(f"tokens={len(ids)} learned={len(learned)}")
            print("LEARNED TEXT:", collator.tokenizer.decode(learned))
        return 0

    if args.dtype == "bf16" or (args.dtype == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=dtype,
    )
    model = load_auto_model_class().from_pretrained(
        args.model,
        quantization_config=bnb,
        dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=not args.no_gradient_checkpointing
    )

    lora = LoraConfig(
        **supported_kwargs(
            LoraConfig,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=None if args.no_modules_to_save else ["lm_head", "embed_tokens"],
            ensure_weight_tying=not args.no_modules_to_save,
        )
    )

    has_val = bool(val_rows)
    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to=args.report_to,
        seed=args.seed,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if has_val else None,
        peft_config=lora,
        processing_class=processor,
        data_collator=collator,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"saved adapter -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
