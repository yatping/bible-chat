#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the full-precision base model for GGUF conversion.

Loads the base in bf16 (NOT 4-bit: merging into quantised weights loses precision), applies the
adapter with PeftModel.from_pretrained, merge_and_unload(), and saves safetensors shards of at
most 2 GB together with the processor (tokenizer + chat template).

Example:
  python training/scripts/merge_adapter.py \\
      --adapter training/outputs/gemma4-e2b-bible-qlora/adapter \\
      --out training/outputs/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bible_ft import TRAINING_DIR

DEFAULT_MODEL = "google/gemma-4-E2B-it"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, default=TRAINING_DIR / "outputs" / "merged")
    p.add_argument("--max-shard-size", default="2GB")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", default="auto", help='e.g. "auto" or "cpu"')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    from train_qlora import load_auto_model_class

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = load_auto_model_class().from_pretrained(args.base_model, dtype=dtype, device_map=args.device_map)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True, max_shard_size=args.max_shard_size)
    # Processor from the adapter dir if saved there (same tokenizer/template), else from the base.
    processor_src = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.base_model
    AutoProcessor.from_pretrained(str(processor_src)).save_pretrained(str(args.out))
    print(f"merged model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
