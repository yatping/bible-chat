"""66-book Protestant canon and book-name aliases.

Aliases are normalised with :func:`normalise_alias` (lowercase, dots removed, single spaces).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    index: int  # 1 (Genesis) .. 66 (Revelation)
    osis: str
    usfm: str
    name: str
    testament: str  # "OT" | "NT"
    chapters: int


# (osis, usfm, name, chapters)
_CANON: list[tuple[str, str, str, int]] = [
    ("Gen", "GEN", "Genesis", 50),
    ("Exod", "EXO", "Exodus", 40),
    ("Lev", "LEV", "Leviticus", 27),
    ("Num", "NUM", "Numbers", 36),
    ("Deut", "DEU", "Deuteronomy", 34),
    ("Josh", "JOS", "Joshua", 24),
    ("Judg", "JDG", "Judges", 21),
    ("Ruth", "RUT", "Ruth", 4),
    ("1Sam", "1SA", "1 Samuel", 31),
    ("2Sam", "2SA", "2 Samuel", 24),
    ("1Kgs", "1KI", "1 Kings", 22),
    ("2Kgs", "2KI", "2 Kings", 25),
    ("1Chr", "1CH", "1 Chronicles", 29),
    ("2Chr", "2CH", "2 Chronicles", 36),
    ("Ezra", "EZR", "Ezra", 10),
    ("Neh", "NEH", "Nehemiah", 13),
    ("Esth", "EST", "Esther", 10),
    ("Job", "JOB", "Job", 42),
    ("Ps", "PSA", "Psalms", 150),
    ("Prov", "PRO", "Proverbs", 31),
    ("Eccl", "ECC", "Ecclesiastes", 12),
    ("Song", "SNG", "Song of Solomon", 8),
    ("Isa", "ISA", "Isaiah", 66),
    ("Jer", "JER", "Jeremiah", 52),
    ("Lam", "LAM", "Lamentations", 5),
    ("Ezek", "EZK", "Ezekiel", 48),
    ("Dan", "DAN", "Daniel", 12),
    ("Hos", "HOS", "Hosea", 14),
    ("Joel", "JOL", "Joel", 3),
    ("Amos", "AMO", "Amos", 9),
    ("Obad", "OBA", "Obadiah", 1),
    ("Jonah", "JON", "Jonah", 4),
    ("Mic", "MIC", "Micah", 7),
    ("Nah", "NAM", "Nahum", 3),
    ("Hab", "HAB", "Habakkuk", 3),
    ("Zeph", "ZEP", "Zephaniah", 3),
    ("Hag", "HAG", "Haggai", 2),
    ("Zech", "ZEC", "Zechariah", 14),
    ("Mal", "MAL", "Malachi", 4),
    ("Matt", "MAT", "Matthew", 28),
    ("Mark", "MRK", "Mark", 16),
    ("Luke", "LUK", "Luke", 24),
    ("John", "JHN", "John", 21),
    ("Acts", "ACT", "Acts", 28),
    ("Rom", "ROM", "Romans", 16),
    ("1Cor", "1CO", "1 Corinthians", 16),
    ("2Cor", "2CO", "2 Corinthians", 13),
    ("Gal", "GAL", "Galatians", 6),
    ("Eph", "EPH", "Ephesians", 6),
    ("Phil", "PHP", "Philippians", 4),
    ("Col", "COL", "Colossians", 4),
    ("1Thess", "1TH", "1 Thessalonians", 5),
    ("2Thess", "2TH", "2 Thessalonians", 3),
    ("1Tim", "1TI", "1 Timothy", 6),
    ("2Tim", "2TI", "2 Timothy", 4),
    ("Titus", "TIT", "Titus", 3),
    ("Phlm", "PHM", "Philemon", 1),
    ("Heb", "HEB", "Hebrews", 13),
    ("Jas", "JAS", "James", 5),
    ("1Pet", "1PE", "1 Peter", 5),
    ("2Pet", "2PE", "2 Peter", 3),
    ("1John", "1JN", "1 John", 5),
    ("2John", "2JN", "2 John", 1),
    ("3John", "3JN", "3 John", 1),
    ("Jude", "JUD", "Jude", 1),
    ("Rev", "REV", "Revelation", 22),
]

BOOKS: list[Book] = [
    Book(i + 1, osis, usfm, name, "OT" if i < 39 else "NT", ch)
    for i, (osis, usfm, name, ch) in enumerate(_CANON)
]
BY_INDEX: dict[int, Book] = {b.index: b for b in BOOKS}
BY_USFM: dict[str, Book] = {b.usfm: b for b in BOOKS}

# Extra aliases per book (beyond name / osis / usfm, which are added automatically).
# Numbered books list only the stem; ordinal prefixes are generated.
_EXTRA: dict[str, list[str]] = {
    "Gen": ["ge", "gn", "gen"],
    "Exod": ["ex", "exo", "exod"],
    "Lev": ["le", "lv"],
    "Num": ["nu", "nm", "nb", "numb"],
    "Deut": ["dt", "de", "deu"],
    "Josh": ["jos", "jsh"],
    "Judg": ["jdg", "jg", "jdgs", "judg"],
    "Ruth": ["ru", "rth", "rut"],
    "Esth": ["es", "est", "esth"],
    "Job": ["jb"],
    "Ps": ["ps", "psa", "psalm", "psalms", "pss", "psm", "pslm"],
    "Prov": ["pr", "prv", "pro", "prov"],
    "Eccl": ["ec", "ecc", "eccl", "eccles", "qoheleth", "ecclesiastes"],
    "Song": ["song", "song of songs", "song of solomon", "sos", "so", "canticles",
             "canticle of canticles", "sg", "sng", "cant"],
    "Isa": ["is", "isa"],
    "Jer": ["je", "jr", "jer"],
    "Lam": ["la", "lam"],
    "Ezek": ["eze", "ezk", "ezek"],
    "Dan": ["da", "dn"],
    "Hos": ["ho"],
    "Joel": ["jl", "joe"],
    "Amos": ["am"],
    "Obad": ["ob", "oba", "obad"],
    "Jonah": ["jnh", "jon"],
    "Mic": ["mc"],
    "Nah": ["na"],
    "Hab": ["hb"],
    "Zeph": ["zep", "zp"],
    "Hag": ["hg"],
    "Zech": ["zec", "zc", "zech"],
    "Mal": ["ml"],
    "Matt": ["mt", "mat", "matt"],
    "Mark": ["mk", "mr", "mrk", "mar"],
    "Luke": ["lk", "lu", "luk"],
    "John": ["jn", "jhn", "joh"],
    "Acts": ["ac", "act", "acts of the apostles"],
    "Rom": ["ro", "rm"],
    "Gal": ["ga"],
    "Eph": ["ephes"],
    "Phil": ["php", "pp", "phil", "philip"],
    "Col": ["co"],
    "Titus": ["ti", "tit"],
    "Phlm": ["phm", "philem", "phlm"],
    "Heb": ["he"],
    "Jas": ["jm", "jas", "jam", "ja"],
    "Jude": ["jud", "jd"],
    "Rev": ["re", "rv", "rev", "revelations", "apocalypse", "revelation of john",
            "the revelation", "revelation to john"],
}

# Stems for numbered books: osis -> list of stems ("corinthians", "cor", ...).
_NUMBERED_STEMS: dict[str, list[str]] = {
    "Sam": ["samuel", "sam", "sa", "sm"],
    "Kgs": ["kings", "kgs", "ki", "kin", "kg"],
    "Chr": ["chronicles", "chron", "chr", "ch"],
    "Cor": ["corinthians", "cor", "co"],
    "Thess": ["thessalonians", "thess", "thes", "th"],
    "Tim": ["timothy", "tim", "ti", "tm"],
    "Pet": ["peter", "pet", "pe", "pt"],
    "John": ["john", "jn", "jhn", "joh", "jo"],
}
_ORDINALS: dict[int, list[str]] = {
    1: ["1", "i", "first", "1st"],
    2: ["2", "ii", "second", "2nd"],
    3: ["3", "iii", "third", "3rd"],
}


def normalise_alias(text: str) -> str:
    """Lowercase, drop dots, collapse whitespace."""
    text = text.lower().replace(".", " ")
    return re.sub(r"\s+", " ", text).strip()


def _build_aliases() -> dict[str, int]:
    by_osis = {b.osis: b for b in BOOKS}
    aliases: dict[str, int] = {}

    def add(alias: str, book: Book) -> None:
        key = normalise_alias(alias)
        if key and key not in aliases:
            aliases[key] = book.index

    for book in BOOKS:
        add(book.name, book)
        add(book.osis, book)
        add(book.usfm, book)
        for extra in _EXTRA.get(book.osis, []):
            add(extra, book)
        m = re.fullmatch(r"([123])(\w+)", book.osis)
        if m:
            num, stem = int(m.group(1)), m.group(2)
            for ordinal in _ORDINALS[num]:
                for s in _NUMBERED_STEMS[stem]:
                    add(f"{ordinal} {s}", book)
                    if ordinal.isdigit():
                        add(f"{ordinal}{s}", book)
    # Ambiguous short forms resolved to the most common reading.
    aliases["jo"] = by_osis["John"].index
    return aliases


ALIASES: dict[str, int] = _build_aliases()


def book_by_alias(alias: str) -> Book | None:
    idx = ALIASES.get(normalise_alias(alias))
    return BY_INDEX[idx] if idx is not None else None
