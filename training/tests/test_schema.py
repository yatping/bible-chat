import copy
import re

import pytest

from bible_ft import TRAINING_DIR
from bible_ft.augment import expand_variants
from bible_ft.jsonl import read_seed_dir
from bible_ft.schema import CATEGORIES, is_rag, prompt_messages, to_messages, validate_dataset, validate_record

ALL_SEEDS = read_seed_dir(TRAINING_DIR / "data" / "seed")
SEEDS = [r for r in ALL_SEEDS if is_rag(r)]


def test_seed_count_and_coverage():
    assert 40 <= len(SEEDS) <= 80
    assert {r["category"] for r in SEEDS} == CATEGORIES


def test_all_seeds_valid():
    assert validate_dataset(ALL_SEEDS, allow_variants=True) == {}


@pytest.mark.parametrize("record", SEEDS, ids=lambda r: r["id"])
def test_answer_quotes_come_from_sources(record):
    """Every “quoted” span in an answer must appear verbatim (modulo punctuation) in SOURCES or the question."""

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[“”‘’\"'.,;:!?—…]", " ", text)).strip().lower()

    for rec in expand_variants(record):
        haystack = norm(" ".join(s["text"] for s in rec["inputs"]["sources"]) + " " + rec["inputs"]["question"])
        for quote in re.findall(r"“([^”]+)”", rec["answer"]):
            assert all(norm(part) in haystack for part in quote.split("…") if norm(part)), quote


def test_to_messages_matches_runtime_prompt_plus_answer():
    record = SEEDS[0]
    messages = to_messages(record)
    assert messages[:-1] == prompt_messages(record)
    assert messages[-1] == {"role": "assistant", "content": record["answer"]}
    assert messages[0]["role"] == "system"


def _base():
    return copy.deepcopy(next(r for r in SEEDS if r["id"] == "seed-direct-first-day"))


def test_rejects_out_of_range_citation():
    r = _base()
    r["answer"] = "God made light [S2]."
    assert any("not in SOURCES" in e for e in validate_record(r))


def test_rejects_grouped_citation():
    r = _base()
    r["answer"] = "God made light [S1, S1]."
    assert any("grouped" in e for e in validate_record(r))


def test_rejects_refusal_with_citation():
    r = _base()
    r["tags"] = ["refusal"]
    r["answer"] = "The supplied sources do not cover this [S1]."
    assert any("must not cite" in e for e in validate_record(r))


def test_rejects_undetectable_refusal():
    r = _base()
    r["tags"] = ["refusal"]
    r["answer"] = "Hmm, hard to say."
    assert any("not detected" in e for e in validate_record(r))


def test_rejects_uncited_answer_and_missing_evidence_without_tag():
    r = _base()
    r["answer"] = "God made light."
    assert any("cite at least one" in e for e in validate_record(r))
    r["category"] = "missing_evidence"
    assert any("tagged 'refusal'" in e for e in validate_record(r))


def test_rejects_bad_history_and_unknown_keys():
    r = _base()
    r["inputs"]["recent_messages"] = [{"role": "user", "content": "hi"}]
    r["extra"] = 1
    errors = validate_record(r)
    assert any("unknown keys" in e for e in errors)
    r.pop("extra")
    assert any("end with an assistant" in e for e in validate_record(r))


def test_variants_only_allowed_in_seed_mode():
    r = next(x for x in SEEDS if "variants" in x)
    assert validate_record(r, allow_variants=True) == []
    assert validate_record(r) != []


CONTENT_DB = TRAINING_DIR.parent / "content" / "build" / "bible.sqlite"


@pytest.mark.skipif(not CONTENT_DB.exists(), reason="content DB not built (optional check)")
def test_seed_scripture_matches_content_db():
    """Seed scripture sources should be what the app retrieves at runtime (same WEB edition)."""
    import sqlite3

    con = sqlite3.connect(f"file:{CONTENT_DB}?mode=ro", uri=True)
    aliases = dict(con.execute("SELECT alias, book_index FROM book_aliases"))
    mismatches = []
    for record in SEEDS:
        for src in record["inputs"]["sources"]:
            m = re.fullmatch(r"(.+?) (\d+):(\d+)(?:-(\d+))?", src["reference"])
            if src["type"] != "scripture" or not m:
                continue
            book = aliases.get(m.group(1).lower()) or aliases.get(m.group(1).lower().rstrip("s"))
            chapter, v1 = int(m.group(2)), int(m.group(3))
            v2 = int(m.group(4) or v1)
            rows = con.execute(
                "SELECT text FROM verses WHERE book_index=? AND chapter=? AND verse BETWEEN ? AND ? ORDER BY verse",
                (book, chapter, v1, v2),
            ).fetchall()
            if " ".join(r[0] for r in rows) != src["text"]:
                mismatches.append(src["reference"])
    con.close()
    assert sorted(set(mismatches)) == []
