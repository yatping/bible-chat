import json

import pytest

from bible_ft import SHARED_DIR
from bible_ft.citations import extract_citations, normalise_citations, renumber_citations, validate_citations

GOLDEN = json.loads((SHARED_DIR / "fixtures" / "citation_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_citation_golden(case):
    result = validate_citations(case["text"], case["source_count"])
    assert result.text == case["expected_text"]
    assert result.cited == case["cited"]
    assert result.removed == case["removed"]


def test_normalise_variants():
    assert normalise_citations("x [S1,S3] y [ S2 - S4 ] z") == "x [S1][S3] y [S2][S3][S4] z"
    assert normalise_citations("[S2–S3]") == "[S2][S3]"
    assert normalise_citations("[S1, S3-S4]") == "[S1][S3][S4]"


def test_removal_strips_tabs_and_spaces_only():
    assert validate_citations("a\t [S5]\nb [S1]", 1).text == "a\nb [S1]"


def test_extract_and_renumber():
    assert extract_citations("a [S2] b [S1][S2]") == ["S2", "S1"]
    assert renumber_citations("[S1] and [S2][S1]", {1: 2, 2: 3}) == "[S2] and [S3][S2]"
