import pytest

from bible_content.books import ALIASES, BOOKS, book_by_alias
from bible_content.references import WHOLE_CHAPTER, Reference, parse_reference, parse_references


def test_canon():
    assert len(BOOKS) == 66 and BOOKS[0].name == "Genesis" and BOOKS[-1].usfm == "REV"
    assert sum(b.chapters for b in BOOKS) == 1189
    assert [b.testament for b in BOOKS].count("OT") == 39


@pytest.mark.parametrize("alias,name", [
    ("mt", "Matthew"), ("matt", "Matthew"), ("Matt.", "Matthew"), ("1 cor", "1 Corinthians"),
    ("1cor", "1 Corinthians"), ("i corinthians", "1 Corinthians"), ("first corinthians", "1 Corinthians"),
    ("song of songs", "Song of Solomon"), ("canticles", "Song of Solomon"), ("psalm", "Psalms"),
    ("psalms", "Psalms"), ("ps", "Psalms"), ("rev", "Revelation"), ("revelations", "Revelation"),
    ("3 jn", "3 John"), ("phlm", "Philemon"), ("ii kings", "2 Kings"), ("jn", "John"),
])
def test_aliases(alias, name):
    book = book_by_alias(alias)
    assert book is not None and book.name == name


def test_aliases_normalised():
    assert all(a == a.lower().strip() and "." not in a and "  " not in a for a in ALIASES)


@pytest.mark.parametrize("text,expected", [
    ("Matthew 18:22", (40, 18, 22, 18, 22)),
    ("Mt 18", (40, 18, 1, 18, WHOLE_CHAPTER)),
    ("1 Cor 13:4-7", (46, 13, 4, 13, 7)),
    ("John 3:16-4:2", (43, 3, 16, 4, 2)),
    ("Ps 23", (19, 23, 1, 23, WHOLE_CHAPTER)),
    ("Psalm 23:1-6", (19, 23, 1, 23, 6)),
    ("Genesis 1:1-2:3", (1, 1, 1, 2, 3)),
    ("What does Rom. 8:28 mean?", (45, 8, 28, 8, 28)),
    ("Jude 3", (65, 1, 3, 1, 3)),
    ("Explain 1 John 4:8", (62, 4, 8, 4, 8)),
    ("What is 2 Tim 3:16 about?", (55, 3, 16, 3, 16)),
    ("Genesis 1-3", (1, 1, 1, 3, WHOLE_CHAPTER)),
    ("Song of Songs 2:1", (22, 2, 1, 2, 1)),
])
def test_parse_reference(text, expected):
    assert parse_reference(text) == Reference(*expected)


@pytest.mark.parametrize("text", [
    "Jesus is 12 years old", "my job 3 times a week", "who is it", "Matthew 29:1", "Genesis",
])
def test_no_false_positives(text):
    assert parse_references(text) == []


def test_multiple_references_in_order():
    refs = parse_references("Compare Is 53:5 with 1 Pet 2:24 and Is 53:5 again")
    assert refs == [Reference(23, 53, 5, 53, 5), Reference(60, 2, 24, 2, 24)]


def test_overlap():
    r = Reference(40, 18, 21, 18, 35)
    assert r.overlaps(40, 18, 15, 18, 21)
    assert r.overlaps(40, 18, 30, 19, 2)
    assert not r.overlaps(40, 18, 1, 18, 20)
    assert not r.overlaps(41, 18, 21, 18, 35)
