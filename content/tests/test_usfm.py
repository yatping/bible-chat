from bible_content.build import load_usfm
from bible_content.usfm import parse_usfm_file


def _verse(books, book_index, chapter, verse):
    book = next(b for b in books if b.book_index == book_index)
    return next(v for v in book.verses if v.chapter == chapter and v.verse == verse)


def test_only_canonical_books_loaded(parsed_books, usfm_dir):
    assert [b.usfm for b in parsed_books] == ["PSA", "MAT"]  # canon order, FRT skipped
    assert parse_usfm_file(usfm_dir / "00-FRTtest.usfm") is None


def test_word_markup_and_footnotes_stripped(parsed_books):
    v = _verse(parsed_books, 40, 17, 24)
    assert v.text == ("When they had come to Capernaum, those who collected the didrachma coins "
                      "came to Peter.")
    for book in parsed_books:
        for verse in book.verses:
            assert "\\" not in verse.text and "|" not in verse.text and "strong=" not in verse.text
            assert verse.text == verse.text.strip() and "  " not in verse.text


def test_cross_reference_note_and_wj_removed(parsed_books):
    v = _verse(parsed_books, 40, 18, 6)
    assert v.text.endswith("Woe to the world because of occasions of stumbling!")
    assert "Mark 9:42" not in v.text


def test_verse_range(parsed_books):
    v = _verse(parsed_books, 40, 18, 6)
    assert v.verse_end == 7


def test_headings_attach_to_next_verse(parsed_books):
    assert _verse(parsed_books, 40, 17, 24).heading == "The temple tax"
    assert _verse(parsed_books, 40, 18, 21).heading == "The parable of the unforgiving servant"
    assert _verse(parsed_books, 40, 18, 22).heading is None
    assert _verse(parsed_books, 40, 18, 1).heading is None


def test_poetry_lines_joined_and_selah_kept(parsed_books):
    v = _verse(parsed_books, 19, 1, 1)
    assert v.text == ("Blessed is the man who doesn’t walk in the counsel of the wicked, nor stand on "
                      "the path of sinners, nor sit in the seat of scoffers;")
    assert _verse(parsed_books, 19, 3, 2).text.endswith("“There is no help for him in God.” Selah.")
    assert _verse(parsed_books, 19, 2, 2).heading is None  # \ms1 / \d are not section headings


def test_superscription_and_titles_excluded(parsed_books):
    assert _verse(parsed_books, 19, 3, 1).text == "Yahweh, how my adversaries have increased!"
    assert _verse(parsed_books, 19, 23, 1).text == "Yahweh is my shepherd; I shall lack nothing."


def test_paragraph_starts(parsed_books):
    assert _verse(parsed_books, 40, 18, 1).para_start  # chapter start
    assert _verse(parsed_books, 40, 18, 22).para_start  # \p
    assert not _verse(parsed_books, 40, 18, 23).para_start
    assert _verse(parsed_books, 19, 3, 3).para_start  # \b stanza break
    assert not _verse(parsed_books, 19, 3, 2).para_start
