import json

import pytest

from bible_content.fts import build_fts_query
from bible_content.paths import SHARED_DIR
from bible_content.rrf import RRF_K, WEIGHT_LEXICAL, WEIGHT_REFERENCE, WEIGHT_SEMANTIC, rrf_fuse

FTS = json.loads((SHARED_DIR / "fixtures" / "fts_query_golden.json").read_text(encoding="utf-8"))
RRF = json.loads((SHARED_DIR / "fixtures" / "rrf_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FTS["cases"], ids=lambda c: c["query"])
def test_fts_query_golden(case):
    assert build_fts_query(case["query"]) == case["expected"]


def test_rrf_constants_match_golden():
    assert RRF["k"] == RRF_K
    weights = {lst["name"]: lst["weight"] for lst in RRF["cases"][0]["lists"]}
    assert weights == {"reference": WEIGHT_REFERENCE, "semantic": WEIGHT_SEMANTIC, "lexical": WEIGHT_LEXICAL}


@pytest.mark.parametrize("case", RRF["cases"], ids=lambda c: c["name"])
def test_rrf_golden(case):
    lists = [(lst["weight"], lst["ids"]) for lst in case["lists"]]
    assert rrf_fuse(lists, top_k=case["top_k"], k=RRF["k"]) == case["expected"]
