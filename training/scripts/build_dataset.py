#!/usr/bin/env python3
"""Expand hand-written seeds into the fine-tuning dataset.

Steps (deterministic for a given --seed):
  1. Load and validate ``data/seed/*.jsonl`` (hard failure on any invalid seed).
  2. Materialise seed memory/format ``variants``.
  3. Optionally merge teacher drafts (only ``needs_review: false`` unless --include-unreviewed).
  4. Negative examples: each answerable question paired with unrelated sources (or none)
     -> refusal answer that cites nothing.
  5. Distractor augmentation: insert irrelevant sources, renumber the answer's [S#] citations.
     (Steps 4-5 apply to rag_answer records only; memory_extraction / summarization seeds pass through.)
  6. Dedup on rendered messages, validate every record, write dataset + stats.

The distractor/negative pool comes from the seeds and, if present, scripture chunks from
``content/build/bible.sqlite``. Held-out test books are never used as pool sources.

Example:
  python training/scripts/build_dataset.py --seed 13
  python training/scripts/build_dataset.py --teacher-drafts training/outputs/teacher_drafts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from bible_ft import REPO_ROOT, TRAINING_DIR
from bible_ft.augment import PoolSource, add_distractors, build_pool, dedup, expand_variants, make_negative
from bible_ft.jsonl import read_jsonl, read_seed_dir, write_jsonl
from bible_ft.schema import is_rag, is_refusal_record, task_of, validate_dataset, validate_record
from bible_ft.split import DEFAULT_TEST_BOOKS

Record = dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", type=Path, default=TRAINING_DIR / "data" / "seed")
    p.add_argument("--content-db", type=Path, default=REPO_ROOT / "content" / "build" / "bible.sqlite",
                   help="optional content DB for extra distractor chunks (skipped if missing)")
    p.add_argument("--no-content-db", action="store_true", help="never read the content DB")
    p.add_argument("--db-pool-size", type=int, default=400, help="scripture chunks sampled from the DB")
    p.add_argument("--teacher-drafts", type=Path, action="append", default=[],
                   help="JSONL from synthesize_with_teacher.py (repeatable)")
    p.add_argument("--include-unreviewed", action="store_true",
                   help="also include teacher drafts still marked needs_review=true")
    p.add_argument("--negatives-per-record", type=int, default=1)
    p.add_argument("--distractors-per-record", type=int, default=2)
    p.add_argument("--max-distractors", type=int, default=2, help="max irrelevant sources inserted per example")
    p.add_argument("--exclude-pool-books", default=",".join(DEFAULT_TEST_BOOKS),
                   help="comma-separated books never used as distractor sources (held-out test books)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=Path, default=TRAINING_DIR / "outputs" / "dataset.jsonl")
    p.add_argument("--stats", type=Path, default=None, help="default: <out stem>_stats.json")
    return p.parse_args(argv)


def load_db_pool(db_path: Path, size: int, exclude_books: set[str], rng: random.Random) -> list[PoolSource]:
    """Sample short scripture chunks from the content DB as extra distractors."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT c.reference, c.text, b.name FROM chunks c JOIN books b ON b.book_index = c.book_index "
            "WHERE c.source_type = 'scripture' AND c.token_estimate <= 300 ORDER BY c.id"
        ).fetchall()
    finally:
        con.close()
    rows = [r for r in rows if r[2] not in exclude_books]
    picked = rng.sample(rows, min(size, len(rows)))
    return [
        PoolSource(source={"type": "scripture", "reference": ref, "text": text}, book=book, topics=frozenset())
        for ref, text, book in picked
    ]


def load_teacher_drafts(paths: list[Path], include_unreviewed: bool) -> tuple[list[Record], Counter[str]]:
    kept: list[Record] = []
    counts: Counter[str] = Counter()
    for path in paths:
        for record in read_jsonl(path):
            counts["read"] += 1
            if record.get("needs_review", True) and not include_unreviewed:
                counts["skipped_unreviewed"] += 1
                continue
            if validate_record(record):
                counts["dropped_invalid"] += 1
                continue
            kept.append(record)
    counts["kept"] = len(kept)
    return kept, counts


def build(args: argparse.Namespace) -> tuple[list[Record], dict[str, Any]]:
    rng = random.Random(args.seed)
    excluded = {b.strip() for b in args.exclude_pool_books.split(",") if b.strip()}

    seeds = read_seed_dir(args.seed_dir)
    if not seeds:
        raise SystemExit(f"no seed records found in {args.seed_dir}")
    failures = validate_dataset(seeds, allow_variants=True)
    if failures:
        for rid, errs in failures.items():
            print(f"INVALID seed {rid}: {'; '.join(errs)}", file=sys.stderr)
        raise SystemExit(1)

    base: list[Record] = [r for seed in seeds for r in expand_variants(seed)]
    teacher, teacher_counts = load_teacher_drafts(args.teacher_drafts, args.include_unreviewed)
    base.extend(teacher)

    pool = build_pool(base, exclude_books=excluded)
    db_used = False
    if not args.no_content_db and args.content_db.exists():
        pool.extend(load_db_pool(args.content_db, args.db_pool_size, excluded, rng))
        db_used = True

    negatives: list[Record] = []
    distracted: list[Record] = []
    for record in base:
        if not is_rag(record) or is_refusal_record(record):
            continue
        for _ in range(args.negatives_per_record):
            if (neg := make_negative(record, pool, rng)) is not None:
                negatives.append(neg)
        for _ in range(args.distractors_per_record):
            if (dis := add_distractors(record, pool, rng, args.max_distractors)) is not None:
                distracted.append(dis)

    records = dedup(base + negatives + distracted)
    failures = validate_dataset(records)
    if failures:
        for rid, errs in failures.items():
            print(f"INVALID generated record {rid}: {'; '.join(errs)}", file=sys.stderr)
        raise SystemExit(1)

    stats = {
        "seed": args.seed,
        "seed_records": len(seeds),
        "base_records": len(base),
        "teacher": dict(teacher_counts),
        "content_db_used": db_used,
        "pool_size": len(pool),
        "negatives": len(negatives),
        "distractor_variants": len(distracted),
        "total": len(records),
        "refusals": sum(is_refusal_record(r) for r in records),
        "by_task": dict(sorted(Counter(task_of(r) for r in records).items())),
        "by_category": dict(sorted(Counter(r["category"] for r in records).items())),
        "by_book": dict(sorted(Counter(r["book"] for r in records if is_rag(r)).items())),
        "by_source_count": dict(sorted(Counter(len(r["inputs"]["sources"]) for r in records if is_rag(r)).items())),
        "by_augmentation": dict(sorted(Counter(
            next((t for t in r["tags"] if t.startswith("augmented:")), "original") for r in records
        ).items())),
    }
    return records, stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records, stats = build(args)
    write_jsonl(args.out, records)
    stats_path = args.stats or args.out.with_name(args.out.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
