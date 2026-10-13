import pytest

from bible_ft.metrics import EvalItem, check_format, compute_metrics, has_refusal_phrase, is_refusal


@pytest.mark.parametrize("text", [
    "The supplied sources do not cover this question.",
    "I can't answer that from the provided sources.",
    "There is not enough information in the sources to answer.",
    "No sources were retrieved for this question.",
    "The passages supplied (Ruth 1:16) do not address it.",
    # untuned Gemma 4 phrasings observed in the MLX base-model eval
    "I do not have the specific information in my sources to determine which city Paul was in.",
    "I do not have the text for Philippians 4:6-7 in the provided sources.",
    "I need the text of Micah 6:8 to tell you what it means.",
])
def test_refusal_phrases(text):
    assert has_refusal_phrase(text)
    assert is_refusal(text, 2)


def test_refusal_with_citation_is_not_refusal():
    text = "The sources do not say her name, but she entered the ship [S1]."
    assert has_refusal_phrase(text)
    assert not is_refusal(text, 1)


def test_plain_answer_is_not_refusal():
    assert not has_refusal_phrase("Jesus wept [S1].")
    assert not has_refusal_phrase("The text refers to a huge fish, but it does not specify a whale [S1].")


def test_format_checks():
    assert check_format("Short answer [S1].", "short")
    assert not check_format(" ".join(["word"] * 120), "short")
    assert check_format("- a [S1]\n- b [S2]", "bullets")
    assert not check_format("one paragraph only", "bullets")
    assert check_format("Key points:\n- a\n- b", "study_notes")
    assert check_format(" ".join(["word"] * 130), "detailed")
    assert check_format("Pray instead of worrying, and God's peace will guard you [S1].", "one_sentence")
    assert not check_format("First sentence. Second sentence.", "one_sentence")


def test_compute_metrics():
    items = [
        # answerable, valid citation of the gold source
        EvalItem("Jesus wept [S1].", 2, False, ("S1",)),
        # answerable, cites a distractor and an invalid id
        EvalItem("He said so [S2][S7].", 2, False, ("S1",), "short"),
        # answerable but falsely refused
        EvalItem("The supplied sources do not cover this.", 1, False, ("S1",)),
        # refusal expected, correctly refused
        EvalItem("I can't answer this from the supplied sources.", 1, True),
        # refusal expected, model answered with a citation
        EvalItem("It was a whale [S1].", 1, True, (), "bullets"),
    ]
    m = compute_metrics(items)
    assert m.n == 5 and m.n_answerable == 3 and m.n_refusal_expected == 2
    assert m.citation_validity_rate == pytest.approx(3 / 4)       # S1, S2, S7, S1 -> S7 invalid
    assert m.unsupported_citation_rate == pytest.approx(2 / 3)    # S2 (distractor) + S1 on refusal item
    assert m.answers_with_invalid_citation_rate == pytest.approx(1 / 5)
    assert m.correct_refusal_rate == pytest.approx(1 / 2)
    assert m.false_refusal_rate == pytest.approx(1 / 3)
    assert m.citation_coverage == pytest.approx(2 / 3)
    assert m.format_adherence == pytest.approx(1 / 2)
    assert m.n_format_items == 2


def test_empty_metrics_are_none():
    m = compute_metrics([])
    assert m.citation_validity_rate is None and m.false_refusal_rate is None
