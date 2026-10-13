"""Find Bible references in free text: "Matthew 18:22", "Mt 18", "1 Cor 13:4-7", "John 3:16-4:2", "Ps 23".

A parsed :class:`Reference` is an inclusive span ``(chapter, verse_start) .. (chapter_end, verse_end)``
within one book. A chapter-only reference spans the whole chapter (``verse_end = WHOLE_CHAPTER``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .books import ALIASES, BY_INDEX

WHOLE_CHAPTER = 999

# Aliases that are also common English words: only accepted when written capitalised
# ("Job 38", "Is 53") or with an explicit chapter:verse ("is 53:5").
_RISKY = {
    "is", "am", "as", "he", "so", "do", "re", "la", "da", "ho", "ac", "co", "ex", "job", "acts",
    "mark", "song", "numbers", "ruth", "jude", "amos", "joel", "ob", "na", "es", "ti", "de", "ge",
    "mr", "pr", "jo", "sg", "pp", "ml", "jd", "jm", "lu", "le", "ja", "ro", "ga", "dt", "hg", "mc",
    "jl", "am", "nb", "nu", "ru", "hb", "rv", "lk", "mk", "jn", "eze", "dan", "mal", "mic", "hag",
    "nah", "jer", "lam", "col", "tit", "jam", "jas", "pro", "act", "rev", "est", "joe", "jon",
}


@dataclass(frozen=True)
class Reference:
    book_index: int
    chapter: int
    verse_start: int
    chapter_end: int
    verse_end: int

    def overlaps(self, book_index: int, chapter: int, verse_start: int,
                 chapter_end: int, verse_end: int) -> bool:
        return (book_index == self.book_index
                and (chapter, verse_start) <= (self.chapter_end, self.verse_end)
                and (chapter_end, verse_end) >= (self.chapter, self.verse_start))


def _alias_pattern(alias: str) -> str:
    words = alias.split(" ")
    # "1cor" style aliases: allow an optional space/dot between the digit prefix and the stem.
    return r"[\s.]+".join(re.escape(w) for w in words) + r"\.?"


_ALIAS_ORDER = sorted(ALIASES, key=len, reverse=True)
_REF_RE = re.compile(
    r"(?<![\w])(?P<book>" + "|".join(_alias_pattern(a) for a in _ALIAS_ORDER) + r")"
    r"\s*(?P<c1>\d{1,3})"
    r"(?:\s*[:.]\s*(?P<v1>\d{1,3})[a-c]?)?"
    r"(?:\s*[-–—]\s*(?P<n2>\d{1,3})(?:\s*[:.]\s*(?P<v2>\d{1,3})[a-c]?)?)?"
    r"(?![\w:])",
    re.IGNORECASE,
)


def _norm(book_text: str) -> str:
    return re.sub(r"\s+", " ", book_text.lower().replace(".", " ")).strip()


def parse_references(text: str) -> list[Reference]:
    """All references found in ``text``, in order of appearance (duplicates removed)."""
    found: list[Reference] = []
    pos = 0
    while (m := _REF_RE.search(text, pos)) is not None:
        ref = _accept(m)
        # A rejected match ("is 2" in "is 2 Tim 3:16") must not hide an overlapping later one.
        pos = m.end() if ref is not None else m.start() + 1
        if ref is not None and ref not in found:
            found.append(ref)
    return found


def _accept(m: re.Match[str]) -> Reference | None:
    book_text = m.group("book")
    alias = _norm(book_text)
    book_index = ALIASES.get(alias) or ALIASES.get(alias.replace(" ", ""))
    if book_index is None:
        return None
    if alias in _RISKY and not (book_text[0].isupper() or m.group("v1")):
        return None
    return _build(book_index, m)


def _build(book_index: int, m: re.Match[str]) -> Reference | None:
    book = BY_INDEX[book_index]
    c1 = int(m.group("c1"))
    v1 = int(m.group("v1")) if m.group("v1") else None
    n2 = int(m.group("n2")) if m.group("n2") else None
    v2 = int(m.group("v2")) if m.group("v2") else None

    if book.chapters == 1 and v1 is None and c1 != 1:
        # "Jude 3" means Jude 1:3 (and "Jude 3-5" means verses 3-5).
        c1, v1 = 1, c1
    if v1 is None:
        # Chapter or chapter range: "Ps 23", "Genesis 1-3" ("Genesis 1-2:3" also accepted).
        c2 = n2 if n2 is not None else c1
        ref = Reference(book_index, c1, 1, c2, v2 if v2 is not None else WHOLE_CHAPTER)
    elif n2 is None:
        ref = Reference(book_index, c1, v1, c1, v1)
    elif v2 is None:
        ref = Reference(book_index, c1, v1, c1, n2)  # "13:4-7"
    else:
        ref = Reference(book_index, c1, v1, n2, v2)  # "3:16-4:2"
    if not (1 <= ref.chapter <= ref.chapter_end <= book.chapters):
        return None
    if (ref.chapter, ref.verse_start) > (ref.chapter_end, ref.verse_end) or ref.verse_start < 1:
        return None
    return ref


def parse_reference(text: str) -> Reference | None:
    """Parse a single reference string such as "Matthew 18:21-35" (first match)."""
    refs = parse_references(text)
    return refs[0] if refs else None
