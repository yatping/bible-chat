"""Minimal but robust USFM parser for scripture text (tuned on ebible.org WEB USFM).

Output is a flat list of :class:`Verse` records with plain text plus structural hints used by
the chunker: the section heading (``\\s``/``\\s1``..) that precedes a verse and whether the
verse starts a new paragraph / stanza.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .books import BY_USFM


@dataclass
class Verse:
    book_index: int
    chapter: int
    verse: int
    text: str
    verse_end: int | None = None  # set for bridged verses such as "\v 1-2"
    heading: str | None = None  # section heading immediately preceding this verse
    para_start: bool = False  # verse begins a paragraph / stanza / chapter


@dataclass
class ParsedBook:
    usfm: str
    book_index: int
    verses: list[Verse]


# Notes (footnotes, endnotes, cross-reference notes) are removed with their content.
_NOTE_RE = re.compile(r"\\(f|fe|ef|x|ex)\s.*?\\\1\*", re.DOTALL)
# Word-level markup: \w word|strong="H123"\w*  (also nested \+w)
_WORD_RE = re.compile(r"\\(\+?)(w|wg|wh|wa)\s+([^\\|]*?)(?:\|[^\\]*?)?\\\1\2\*")
# Remaining character markers (opening consumes one delimiter space, closing has '*').
_CHAR_MARKERS = (
    "add|addpn|bd|bdit|bk|dc|em|fig|it|jmp|k|lit|nd|no|ord|pn|png|qac|qs|qt|rb|rq|sc|sig|sls|"
    "sup|tl|wj|w|wg|wh|wa|ior|iqt"
)
_CHAR_OPEN_RE = re.compile(rf"\\\+?(?:{_CHAR_MARKERS})(?:\s|(?=\\))")
_CHAR_CLOSE_RE = re.compile(r"\\\+?[a-z]+[0-9]*\*")  # any closing marker
_MARKER_RE = re.compile(r"\\([a-z]+[0-9]*)(?=[\s\\]|$)\s?")

_HEADING_MARKERS = re.compile(r"s[1-4]?")
# Markers whose content is not verse text (titles, intros, labels, headings ...).
_NON_TEXT = re.compile(
    r"(id|ide|h[1-3]?|toc[1-3]?|toca[1-3]?|rem|sts|usfm|mt[1-4]?|mte[1-4]?|ms[1-4]?|mr|s[1-4]?|"
    r"sr|r|d|sp|sd[1-4]?|cl|cp|cd|ca|va|vp|imt[1-4]?|is[1-4]?|ip|ipi|im|imi|ipq|imq|ipr|iq[1-4]?|"
    r"ib|ili[1-4]?|iot|io[1-4]?|iex|imte|ie|restore|periph|lit)"
)
# Markers that start a new paragraph / stanza (a natural chunk boundary).
_BREAK = re.compile(r"(p|m|pi[1-4]?|mi|pc|pmo|po|cls|b|li1|lh|ms[1-4]?|d|s[1-4]?|qa|sp)")

_WS_RE = re.compile(r"\s+")
_VERSE_NUM_RE = re.compile(r"(\d+)[a-z]?(?:\s*[-\u2013]\s*(\d+)[a-z]?)?")


def _clean_inline(text: str) -> str:
    text = _NOTE_RE.sub("", text)
    text = _WORD_RE.sub(lambda m: m.group(3), text)
    text = _CHAR_OPEN_RE.sub("", text)
    text = _CHAR_CLOSE_RE.sub("", text)
    return text


def normalise_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def parse_usfm(source: str) -> ParsedBook | None:
    """Parse one USFM book. Returns None if the book is not in the 66-book canon."""
    text = _clean_inline(source)
    parts = _MARKER_RE.split(text)
    # parts = [leading, marker1, content1, marker2, content2, ...]
    book_code: str | None = None
    chapter = 0
    verses: list[Verse] = []
    current: Verse | None = None
    pending_heading: str | None = None
    pending_break = False

    for i in range(1, len(parts), 2):
        marker, content = parts[i], parts[i + 1]
        if marker == "id":
            book_code = content.split()[0].upper() if content.split() else None
            continue
        if marker == "c":
            m = re.match(r"\s*(\d+)", content)
            if m:
                chapter = int(m.group(1))
                current = None
                pending_break = True
            continue
        if marker == "v":
            m = _VERSE_NUM_RE.match(content.strip())
            if not m or chapter == 0:
                continue
            body = content.strip()[m.end():]
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else None
            current = Verse(0, chapter, start, body, verse_end=end if end and end > start else None,
                            heading=pending_heading, para_start=pending_break)
            verses.append(current)
            pending_heading, pending_break = None, False
            continue
        if _HEADING_MARKERS.fullmatch(marker):
            heading = normalise_ws(content)
            if heading and chapter > 0:
                pending_heading = heading
            pending_break = True
            continue
        if _NON_TEXT.fullmatch(marker):
            if _BREAK.fullmatch(marker):
                pending_break = True
            continue
        # Paragraph / poetry / list / unknown markers: content continues the current verse
        # (e.g. a \q2 line). A break marker here only affects the *next* verse.
        if _BREAK.fullmatch(marker):
            pending_break = True
        if current is not None:
            current.text += " " + content

    if book_code is None or book_code not in BY_USFM:
        return None
    book = BY_USFM[book_code]
    kept: list[Verse] = []
    carry_heading: str | None = None
    carry_break = False
    for v in verses:
        v.book_index = book.index
        v.text = normalise_ws(v.text)
        if not v.text:  # e.g. a verse present only as a footnote: carry its structure forward
            carry_heading = v.heading or carry_heading
            carry_break = carry_break or v.para_start
            continue
        v.heading = v.heading or carry_heading
        v.para_start = v.para_start or carry_break
        carry_heading, carry_break = None, False
        kept.append(v)
    return ParsedBook(book_code, book.index, kept)


def parse_usfm_file(path: Path) -> ParsedBook | None:
    return parse_usfm(path.read_text(encoding="utf-8-sig"))
