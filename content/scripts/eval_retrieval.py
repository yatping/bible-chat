#!/usr/bin/env python3
"""Evaluate retrieval on a labelled JSONL set; prints a markdown table.

Each line: {"question": "...", "relevant": ["Matthew 18:21-35", ...]}. A retrieved chunk is a hit
when it overlaps any relevant range. Recall@k = share of questions with at least one hit in the
top k; MRR = mean reciprocal rank of the first hit (0 when none in the ranked list, max 30).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import _bootstrap  # noqa: F401
from bible_content.embeddings import HASH_MODEL, make_embedder
from bible_content.paths import BUILD_DIR, EVAL_PATH
from bible_content.references import Reference, parse_reference
from bible_content.retrieval import LIST_LIMIT, ChunkInfo, HybridRetriever

MODES = ("reference", "lexical", "semantic", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=BUILD_DIR / "bible.sqlite")
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--embedder", choices=["auto", "hash", "embeddinggemma"], default="auto",
                        help="query embedder; auto = from content_meta.embedding_model")
    parser.add_argument("--verbose", action="store_true", help="list misses for the hybrid ranking")
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        meta = dict(conn.execute("SELECT key, value FROM content_meta"))
    backend = args.embedder
    if backend == "auto":
        backend = "hash" if meta["embedding_model"] == HASH_MODEL else "embeddinggemma"
    embedder = make_embedder(backend, int(meta["embedding_dim"]))
    retriever = HybridRetriever(args.db, None, embedder)

    items = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = {m: {"r1": 0.0, "r5": 0.0, "mrr": 0.0} for m in MODES}
    misses: list[str] = []
    for item in items:
        parsed = [parse_reference(r) for r in item["relevant"]]
        relevant = [r for r in parsed if r is not None]
        if len(relevant) != len(parsed):
            raise SystemExit(f"unparseable relevant reference in: {item}")
        lists = {
            "reference": retriever.reference_list(item["question"]),
            "lexical": retriever.lexical_list(item["question"]),
            "semantic": retriever.semantic_list(item["question"]),
        }
        lists["hybrid"] = retriever.fused(item["question"], top_k=LIST_LIMIT)
        for mode, ids in lists.items():
            hits = [rank for rank, cid in enumerate(ids, start=1)
                    if _overlaps(retriever.chunks[cid], relevant)]
            first = hits[0] if hits else None
            stats[mode]["r1"] += first == 1
            stats[mode]["r5"] += first is not None and first <= 5
            stats[mode]["mrr"] += 1.0 / first if first else 0.0
            if mode == "hybrid" and (first is None or first > 5):
                top = [retriever.chunks[c].reference for c in ids[:3]]
                misses.append(f"- rank {first}: {item['question']} -> {item['relevant']} (got {top})")

    n = len(items)
    print(f"Retrieval eval: {n} questions, db={args.db}, embedder={embedder.model_name}\n")
    print("| mode | Recall@1 | Recall@5 | MRR |")
    print("|---|---:|---:|---:|")
    for mode in MODES:
        s = stats[mode]
        print(f"| {mode} | {s['r1'] / n:.3f} | {s['r5'] / n:.3f} | {s['mrr'] / n:.3f} |")
    if args.verbose and misses:
        print("\nHybrid misses (not in top 5):")
        print("\n".join(misses))


def _overlaps(chunk: ChunkInfo, relevant: list[Reference]) -> bool:
    if chunk.source_type != "scripture" or chunk.book_index is None:
        return False
    span = (chunk.book_index, chunk.chapter or 0, chunk.verse_start or 0,
            chunk.chapter_end or 0, chunk.verse_end or 0)
    return any(r.overlaps(*span) for r in relevant)


if __name__ == "__main__":
    main()
