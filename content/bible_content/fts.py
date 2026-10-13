"""FTS5 query construction (CONTRACTS.md §3). Must match shared/fixtures/fts_query_golden.json."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .paths import SHARED_DIR

# [\p{L}\p{N}]+ : Python's \w is letters, digits and underscore; [^\W_] drops the underscore.
_TOKEN_RE = re.compile(r"[^\W_]+")


@lru_cache(maxsize=1)
def stopwords(path: Path = SHARED_DIR / "stopwords.txt") -> frozenset[str]:
    words = (line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(w for w in words if w and not w.startswith("#"))


def fts_tokens(query: str) -> list[str]:
    stop = stopwords()
    seen: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if len(token) < 2 or token in stop or token in seen:
            continue
        seen.append(token)
    return seen


def build_fts_query(query: str) -> str | None:
    """Quoted tokens joined with " OR ", or None when no tokens remain (skip lexical search)."""
    tokens = fts_tokens(query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)
