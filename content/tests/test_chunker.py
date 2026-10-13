import random

import pytest

from bible_content.chunker import MAX_VERSES, MIN_VERSES, chunk_book, chunk_books, format_reference
from bible_content.paths import RAW_DIR
from bible_content.usfm import ParsedBook, Verse


def _units(book, chunk):
    return [v for v in book.verses
            if (chunk.chapter, chunk.verse_start) <= (v.chapter, v.verse)
            <= (chunk.chapter_end, chunk.verse_end)]


def assert_invariants(books, chunks):
    by_book = {b.book_index: b for b in books}
    for book in books:
        mine = [c for c in chunks if c.book_index == book.book_index]
        covered = [v for c in mine for v in _units(book, c)]
        assert covered == book.verses, "every verse covered exactly once, in order"
        for c in mine:
            n = len(_units(book, c))
            assert MIN_VERSES <= n <= MAX_VERSES or len(book.verses) < MIN_VERSES, c.reference
            assert c.token_estimate == -(-len(c.text) // 4)
            assert c.text == " ".join(v.text for v in _units(book, c))
            assert c.reference == format_reference(c.book_index, c.chapter, c.verse_start,
                                                   c.chapter_end, c.verse_end)
    assert all(c.book_index in by_book for c in chunks)


def test_fixture_chunks(parsed_books):
    chunks = chunk_books(parsed_books, "WEB")
    assert_invariants(parsed_books, chunks)
    refs = [c.reference for c in chunks]
    assert refs == [
        "Psalm 1:1-2:2", "Psalm 3:1-8", "Psalm 23:1-6",
        "Matthew 17:24-18:5", "Matthew 18:6-11", "Matthew 18:12-14", "Matthew 18:21-35",
        "Matthew 19:1-9", "Matthew 19:10-20",
    ]
    servant = next(c for c in chunks if c.reference == "Matthew 18:21-35")
    assert servant.title == "The parable of the unforgiving servant"
    assert (servant.chapter, servant.verse_start, servant.chapter_end, servant.verse_end) == (18, 21, 18, 35)
    temple = chunks[3]
    assert (temple.chapter, temple.chapter_end) == (17, 18)
    assert chunks[0].title == "Psalm 1:1-2:2"  # no heading -> reference
    assert next(c for c in chunks if c.reference == "Matthew 18:6-11").verse_end == 11


def _synthetic(chapter_sizes, headings=frozenset(), paragraphs=frozenset()):
    verses = []
    for ch, size in enumerate(chapter_sizes, start=1):
        for v in range(1, size + 1):
            verses.append(Verse(1, ch, v, f"word{ch}x{v}", heading=f"H{ch}.{v}" if (ch, v) in headings else None,
                                para_start=v == 1 or (ch, v) in paragraphs))
    return ParsedBook("GEN", 1, verses)


@pytest.mark.parametrize("seed", range(30))
def test_random_books_invariants(seed):
    rng = random.Random(seed)
    sizes = [rng.randint(1, 60) for _ in range(rng.randint(1, 8))]
    if sum(sizes) < MIN_VERSES:
        sizes.append(5)
    all_verses = [(c, v) for c, n in enumerate(sizes, start=1) for v in range(1, n + 1)]
    headings = frozenset(rng.sample(all_verses, k=rng.randint(0, len(all_verses) // 4)))
    paragraphs = frozenset(rng.sample(all_verses, k=len(all_verses) // 5))
    book = _synthetic(sizes, headings, paragraphs)
    assert_invariants([book], chunk_book(book, "WEB"))


def test_fallback_windows_stay_within_chapter():
    book = _synthetic([40, 30])
    chunks = chunk_book(book, "WEB")
    assert all(c.chapter == c.chapter_end for c in chunks)
    assert all(6 <= c.verse_end - c.verse_start + 1 <= 10 for c in chunks)


def test_split_prefers_paragraph_boundary():
    book = _synthetic([16], paragraphs=frozenset({(1, 7)}))
    assert [c.reference for c in chunk_book(book, "WEB")] == ["Genesis 1:1-6", "Genesis 1:7-16"]


@pytest.mark.skipif(not (RAW_DIR / "engwebp_usfm.zip").exists(), reason="WEB USFM not downloaded")
def test_real_web_invariants():
    from bible_content.build import load_usfm

    books = load_usfm(RAW_DIR / "engwebp_usfm.zip")
    assert len(books) == 66
    assert 31000 < sum(len(b.verses) for b in books) < 31200
    assert_invariants(books, chunk_books(books, "WEB"))
