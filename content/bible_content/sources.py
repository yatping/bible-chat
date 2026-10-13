"""Secondary sources: OpenBible.info cross references and generic commentary / dictionary JSONL."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from .books import BOOKS
from .chunker import Chunk, token_estimate
from .db import CrossReference
from .references import WHOLE_CHAPTER, parse_reference

_OSIS = {b.osis: b.index for b in BOOKS}
_OSIS_VERSE = re.compile(r"(\w+)\.(\d+)\.(\d+)")

COMMENTARY_MIN_TOKENS = 200
COMMENTARY_MAX_TOKENS = 500
COMMENTARY_TARGET_TOKENS = 400


# --- OpenBible.info cross references (CC-BY) --------------------------------------------------

def parse_openbible_xrefs(path: Path, min_votes: int = 0) -> Iterator[CrossReference]:
    """Parse cross_references.txt ("Gen.1.1<TAB>Prov.8.22-Prov.8.30<TAB>59").

    Ranges that leave the start chapter are clamped to the start verse.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(("From Verse", "#")) or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            src = _OSIS_VERSE.fullmatch(parts[0])
            dst = parts[1].split("-")
            start = _OSIS_VERSE.fullmatch(dst[0])
            end = _OSIS_VERSE.fullmatch(dst[-1])
            try:
                votes = int(parts[2])
            except ValueError:
                continue
            if not (src and start and end) or votes < min_votes:
                continue
            fb, tb = _OSIS.get(src.group(1)), _OSIS.get(start.group(1))
            if fb is None or tb is None:
                continue
            to_start = int(start.group(3))
            same_chapter = end.group(1) == start.group(1) and end.group(2) == start.group(2)
            to_end = int(end.group(3)) if same_chapter else to_start
            yield CrossReference(fb, int(src.group(2)), int(src.group(3)),
                                 tb, int(start.group(2)), to_start, max(to_start, to_end), votes)


# --- Commentary / dictionary JSONL -------------------------------------------------------------

def _split_long(text: str) -> list[str]:
    """Split text into pieces of ~COMMENTARY_TARGET_TOKENS (≤ MAX), on paragraph then sentence breaks."""
    if token_estimate(text) <= COMMENTARY_MAX_TOKENS:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if token_estimate(para) <= COMMENTARY_TARGET_TOKENS:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?;])\s+", para) if s)
    pieces: list[str] = []
    current = ""
    for unit in units:
        while token_estimate(unit) > COMMENTARY_MAX_TOKENS:  # pathological sentence: hard split
            cut = COMMENTARY_TARGET_TOKENS * 4
            head, unit = unit[:cut], unit[cut:]
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        candidate = f"{current} {unit}".strip()
        if current and token_estimate(candidate) > COMMENTARY_TARGET_TOKENS:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        if pieces and token_estimate(current) < COMMENTARY_MIN_TOKENS \
                and token_estimate(pieces[-1] + " " + current) <= COMMENTARY_MAX_TOKENS:
            pieces[-1] = pieces[-1] + " " + current
        else:
            pieces.append(current)
    return pieces


def load_commentary_jsonl(path: Path) -> list[Chunk]:
    """Each line: {"source_type": "commentary"|"dictionary", "title", "reference"?, "text"}."""
    chunks: list[Chunk] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        source_type = entry.get("source_type")
        if source_type not in ("commentary", "dictionary"):
            raise ValueError(f"{path}:{lineno}: source_type must be commentary or dictionary")
        title = str(entry["title"]).strip()
        reference = str(entry.get("reference") or "").strip()
        ref = parse_reference(reference) if reference else None
        pieces = _split_long(str(entry["text"]).strip())
        for n, piece in enumerate(pieces, start=1):
            chunks.append(Chunk(
                source_type=source_type,
                title=title if len(pieces) == 1 else f"{title} ({n}/{len(pieces)})",
                reference=reference or title,
                text=" ".join(piece.split()),
                book_index=ref.book_index if ref else None,
                chapter=ref.chapter if ref else None,
                verse_start=ref.verse_start if ref else None,
                chapter_end=ref.chapter_end if ref else None,
                verse_end=(ref.verse_end if ref and ref.verse_end != WHOLE_CHAPTER else None),
            ))
    return chunks
