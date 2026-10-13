"""Pericope chunking of parsed scripture.

Algorithm (per book, never crossing book boundaries):

1. Sections: if the book has section headings, split at headings (a section may run across a
   chapter boundary; the chunk then records ``chapter_end``). Without headings (e.g. the ebible.org
   WEB USFM, which has none) each chapter is a section.
2. Sections shorter than ``MIN_VERSES`` are merged into the previous section (the next one for the
   first section of a book).
3. Sections longer than ``MAX_VERSES`` (heading mode) are split into balanced pieces of
   ``ceil(n / ceil(n / MAX_VERSES))`` target size; chapter sections (no-heading fallback) are split
   into pieces of ~``FALLBACK_TARGET`` verses. Cuts are chosen by dynamic programming and prefer
   paragraph / stanza starts.

References use the singular "Psalm" for the book of Psalms ("Psalm 23:1-6"); every other book uses
its canonical name. Cross-chapter chunks are written "Genesis 1:1-2:3".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .books import BY_INDEX
from .usfm import ParsedBook, Verse

MIN_VERSES = 3
MAX_VERSES = 15
FALLBACK_TARGET = 8
NON_PARAGRAPH_CUT_PENALTY = 20.0


@dataclass
class Chunk:
    source_type: str
    title: str
    reference: str
    text: str
    translation: str | None = None
    book_index: int | None = None
    chapter: int | None = None
    verse_start: int | None = None
    chapter_end: int | None = None
    verse_end: int | None = None

    @property
    def token_estimate(self) -> int:
        return token_estimate(self.text)


def token_estimate(text: str) -> int:
    return math.ceil(len(text) / 4)


def reference_book_name(book_index: int) -> str:
    return "Psalm" if book_index == 19 else BY_INDEX[book_index].name


def format_reference(book_index: int, chapter: int, verse_start: int,
                     chapter_end: int, verse_end: int) -> str:
    name = reference_book_name(book_index)
    if chapter_end != chapter:
        return f"{name} {chapter}:{verse_start}-{chapter_end}:{verse_end}"
    if verse_end != verse_start:
        return f"{name} {chapter}:{verse_start}-{verse_end}"
    return f"{name} {chapter}:{verse_start}"


def _last_verse(v: Verse) -> int:
    return v.verse_end or v.verse


def _sections(verses: list[Verse]) -> tuple[list[list[Verse]], bool]:
    has_headings = any(v.heading for v in verses)
    sections: list[list[Verse]] = []
    for i, v in enumerate(verses):
        if i == 0:
            new = True
        elif has_headings:
            new = v.heading is not None
        else:
            new = v.chapter != verses[i - 1].chapter
        if new:
            sections.append([v])
        else:
            sections[-1].append(v)
    return sections, has_headings


def _merge_small(sections: list[list[Verse]]) -> list[list[Verse]]:
    """Merge each section shorter than MIN_VERSES into a neighbour: prefer a neighbour in the same
    chapter, then the one giving the smaller merged section, then the previous one."""
    merged = [list(s) for s in sections]
    while len(merged) > 1:
        small = next((i for i, s in enumerate(merged) if len(s) < MIN_VERSES), None)
        if small is None:
            break
        sec = merged[small]
        candidates = []
        if small > 0:
            prev = merged[small - 1]
            candidates.append((prev[-1].chapter != sec[0].chapter, len(prev), 0, small - 1))
        if small + 1 < len(merged):
            nxt = merged[small + 1]
            candidates.append((nxt[0].chapter != sec[-1].chapter, len(nxt), 1, small + 1))
        *_, target = min(candidates)
        lo, hi = sorted((small, target))
        merged[lo:hi + 1] = [merged[lo] + merged[hi]]
    return merged


def _split(section: list[Verse], target: float) -> list[list[Verse]]:
    """Split into pieces of MIN..MAX verses, sizes close to ``target``, preferring paragraph cuts."""
    n = len(section)
    inf = float("inf")
    cost = [inf] * (n + 1)
    back = [0] * (n + 1)
    cost[0] = 0.0
    for end in range(MIN_VERSES, n + 1):
        for start in range(max(0, end - MAX_VERSES), end - MIN_VERSES + 1):
            if cost[start] == inf:
                continue
            c = cost[start] + (end - start - target) ** 2
            if start > 0 and not section[start].para_start:
                c += NON_PARAGRAPH_CUT_PENALTY
            if c < cost[end]:
                cost[end], back[end] = c, start
    if cost[n] == inf:  # fewer than MIN_VERSES verses in total: keep as one piece
        return [section]
    pieces: list[list[Verse]] = []
    end = n
    while end > 0:
        start = back[end]
        pieces.append(section[start:end])
        end = start
    return pieces[::-1]


def chunk_book(book: ParsedBook, translation: str) -> list[Chunk]:
    sections, has_headings = _sections(book.verses)
    sections = _merge_small(sections)
    chunks: list[Chunk] = []
    for sec in sections:
        section_heading = next((v.heading for v in sec if v.heading), None)
        if has_headings:
            pieces = [sec] if len(sec) <= MAX_VERSES else _split(sec, len(sec) / math.ceil(len(sec) / MAX_VERSES))
        else:
            pieces = [sec] if len(sec) < 2 * MIN_VERSES else _split(sec, FALLBACK_TARGET)
        for piece in pieces:
            first, last = piece[0], piece[-1]
            ref = format_reference(book.book_index, first.chapter, first.verse, last.chapter, _last_verse(last))
            chunks.append(Chunk(
                source_type="scripture",
                title=section_heading or ref,
                reference=ref,
                text=" ".join(v.text for v in piece),
                translation=translation,
                book_index=book.book_index,
                chapter=first.chapter,
                verse_start=first.verse,
                chapter_end=last.chapter,
                verse_end=_last_verse(last),
            ))
    return chunks


def chunk_books(books: list[ParsedBook], translation: str) -> list[Chunk]:
    return [c for b in books for c in chunk_book(b, translation)]
