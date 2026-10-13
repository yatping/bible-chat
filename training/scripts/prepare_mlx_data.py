#!/usr/bin/env python3
"""Convert the dataset splits into mlx-lm chat format.

Reads ``outputs/splits/{train,val,test}.jsonl`` (records validated by bible_ft.schema) and writes
``outputs/mlx_data/{train,valid,test}.jsonl`` with one ``{"messages": [...]}`` per line. Messages come from
``bible_ft.schema.to_messages``, so they are exactly the runtime prompt plus the gold answer. mlx-lm renders them
with the model's own chat template (thinking defaults to off, matching the device) and, with ``mask_prompt``,
trains on the final assistant turn only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bible_ft import TRAINING_DIR
from bible_ft.jsonl import read_jsonl, write_jsonl
from bible_ft.schema import to_messages

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # our name -> mlx-lm name


def convert(splits_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for ours, theirs in SPLITS.items():
        rows = [{"messages": to_messages(r)} for r in read_jsonl(splits_dir / f"{ours}.jsonl")]
        write_jsonl(out_dir / f"{theirs}.jsonl", rows)
        counts[theirs] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=TRAINING_DIR / "outputs" / "splits")
    p.add_argument("--out", type=Path, default=TRAINING_DIR / "outputs" / "mlx_data")
    args = p.parse_args(argv)
    for name, n in convert(args.splits, args.out).items():
        print(f"{name:5} {n:5d} -> {args.out / (name + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
