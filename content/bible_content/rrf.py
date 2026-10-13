"""Weighted reciprocal rank fusion (CONTRACTS.md §3). Must match shared/fixtures/rrf_golden.json."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60
WEIGHT_REFERENCE = 2.0
WEIGHT_SEMANTIC = 1.0
WEIGHT_LEXICAL = 1.0


def rrf_fuse(lists: Sequence[tuple[float, Sequence[int]]], top_k: int, k: int = RRF_K) -> list[int]:
    """Fuse ranked id lists given as ``(weight, ids)``; ranks are 1-based.

    Scores are accumulated in list order; ties are broken by lower id.
    """
    scores: dict[int, float] = {}
    for weight, ids in lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:top_k]
