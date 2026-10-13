#!/usr/bin/env python3
"""Split dataset.jsonl into train / val / test (target 80/10/10).

- test: every record whose ``book`` is held out (default: Ruth, Jonah, James, Philippians, Micah),
  so evaluation measures behaviour on passages never seen in training;
- val: random question-groups from the remaining records (``--val-fraction`` of the total);
- train: the rest.

Records are grouped by normalised question, so augmentations of one seed never straddle
splits. The script fails if any question leaks across splits.

Example:
  python training/scripts/split_dataset.py --test-books Ruth,Jonah,James,Philippians,Micah

Memory-extraction / summarization records are split randomly (10% test, 10% val by default).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from bible_ft import TRAINING_DIR
from bible_ft.jsonl import read_jsonl, write_jsonl
from bible_ft.schema import is_refusal_record, task_of
from bible_ft.split import DEFAULT_TEST_BOOKS, leaked_questions, split_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=TRAINING_DIR / "outputs" / "dataset.jsonl")
    p.add_argument("--out-dir", type=Path, default=TRAINING_DIR / "outputs" / "splits")
    p.add_argument("--test-books", default=",".join(DEFAULT_TEST_BOOKS), help="comma-separated held-out books")
    p.add_argument("--val-fraction", type=float, default=0.1, help="fraction of ALL records used for validation")
    p.add_argument("--aux-test-fraction", type=float, default=0.1,
                   help="test fraction for memory_extraction / summarization records (random split)")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = list(read_jsonl(args.input))
    splits = split_records(records, args.test_books.split(","), args.val_fraction, args.seed, args.aux_test_fraction)

    leaked = leaked_questions(splits)
    if leaked:
        raise SystemExit(f"question leakage across splits: {sorted(leaked)[:5]}")

    report: dict[str, object] = {"total": len(records), "test_books": args.test_books.split(",")}
    for name, recs in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", recs)
        report[name] = {
            "records": len(recs),
            "fraction": round(len(recs) / max(1, len(records)), 3),
            "refusals": sum(is_refusal_record(r) for r in recs),
            "tasks": dict(sorted(Counter(task_of(r) for r in recs).items())),
            "categories": dict(sorted(Counter(r["category"] for r in recs).items())),
        }
    if not splits["test"]:
        print("WARNING: test split is empty; check --test-books against the dataset's book field")
    print(json.dumps(report, indent=2))
    (args.out_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
