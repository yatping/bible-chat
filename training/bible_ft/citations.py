"""Citation normalisation and validation (shared/CONTRACTS.md §1, post-generation).

1. Grouped citations are normalised: ``[S1, S2]`` -> ``[S1][S2]``, ``[S2-S4]`` -> ``[S2][S3][S4]``.
2. Every ``[S<k>]`` with k outside 1..n is removed together with spaces/tabs immediately before it.
3. ``cited`` / ``removed`` list ids in first-appearance order, de-duplicated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A bracket containing one or more S-ids or S-ranges separated by commas, e.g. "[S1, S2-S4]".
_ITEM = r"S\d+(?:\s*[-–]\s*S?\d+)?"
_GROUP_RE = re.compile(r"\[\s*(" + _ITEM + r"(?:\s*,\s*" + _ITEM + r")*)\s*\]")
_RANGE_RE = re.compile(r"S(\d+)\s*[-–]\s*S?(\d+)")
_SINGLE_RE = re.compile(r"[ \t]*\[S(\d+)\]")
_ID_RE = re.compile(r"\[S(\d+)\]")
_MAX_RANGE = 50


@dataclass(frozen=True)
class CitationResult:
    text: str
    cited: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _expand_item(item: str) -> list[int]:
    item = item.strip()
    m = _RANGE_RE.fullmatch(item)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start <= end and end - start <= _MAX_RANGE:
            return list(range(start, end + 1))
        return [start, end]
    return [int(item[1:])]


def normalise_citations(text: str) -> str:
    """Rewrite grouped/ranged citations into consecutive single ``[S#]`` tags."""

    def repl(match: re.Match[str]) -> str:
        ids: list[int] = []
        for item in match.group(1).split(","):
            ids.extend(_expand_item(item))
        return "".join(f"[S{i}]" for i in ids)

    return _GROUP_RE.sub(repl, text)


def extract_citations(text: str) -> list[str]:
    """All ``S#`` ids in (normalised) text, first-appearance order, de-duplicated."""
    seen: dict[str, None] = {}
    for m in _ID_RE.finditer(normalise_citations(text)):
        seen.setdefault(f"S{int(m.group(1))}", None)
    return list(seen)


def validate_citations(text: str, source_count: int) -> CitationResult:
    """Normalise, strip out-of-range citations and report cited / removed ids."""
    normalised = normalise_citations(text)
    cited: dict[str, None] = {}
    removed: dict[str, None] = {}

    def repl(match: re.Match[str]) -> str:
        k = int(match.group(1))
        sid = f"S{k}"
        if 1 <= k <= source_count:
            cited.setdefault(sid, None)
            return match.group(0)
        removed.setdefault(sid, None)
        return ""

    cleaned = _SINGLE_RE.sub(repl, normalised)
    return CitationResult(text=cleaned, cited=list(cited), removed=list(removed))


def renumber_citations(text: str, mapping: dict[int, int]) -> str:
    """Simultaneously rewrite ``[S<old>]`` -> ``[S<new>]`` (text must already be normalised)."""

    def repl(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return f"[S{mapping.get(old, old)}]"

    return _ID_RE.sub(repl, text)
