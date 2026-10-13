import json
import re

import pytest

import build_dataset
from bible_ft import TRAINING_DIR
from bible_ft.augment import build_pool, make_negative, topics_of
from bible_ft.citations import extract_citations
from bible_ft.jsonl import read_jsonl, read_seed_dir, write_jsonl
from bible_ft.metrics import is_refusal
from bible_ft.schema import validate_record


def seed(rid, book, topic, question, sources, answer, tags=()):
    return {
        "id": rid, "category": "direct_scripture", "book": book,
        "inputs": {"memories": [], "summary": None, "recent_messages": [], "question": question,
                   "sources": [{"type": "scripture", "reference": ref, "text": text} for ref, text in sources]},
        "answer": answer, "tags": [f"topic:{topic}", *tags],
    }


TINY = [
    seed("s-wept", "John", "grief", "What did Jesus do at Lazarus' tomb?", [("John 11:35", "Jesus wept.")],
         "Jesus wept [S1]."),
    seed("s-light", "Genesis", "creation", "What did God create first?",
         [("Genesis 1:1", "In the beginning, God created the heavens and the earth."),
          ("Genesis 1:3", "God said, “Let there be light,” and there was light.")],
         "God created the heavens and the earth [S1], then said “Let there be light” [S2]."),
    seed("s-murder", "Exodus", "commandments", "Which commandment forbids killing?",
         [("Exodus 20:13", "You shall not murder.")], "“You shall not murder” [S1]."),
    seed("s-faith", "Hebrews", "faith", "How does Hebrews define faith?",
         [("Hebrews 11:1", "Now faith is assurance of things hoped for, proof of things not seen.")],
         "Faith is “assurance of things hoped for, proof of things not seen” [S1]."),
    seed("s-word", "Psalms", "scripture", "What is God's word compared to?",
         [("Psalm 119:105", "Your word is a lamp to my feet, and a light for my path.")],
         "A lamp to the feet and a light for the path [S1]."),
    seed("s-ruth", "Ruth", "loyalty", "What did Ruth promise Naomi?",
         [("Ruth 1:16", "where you go, I will go")], "She promised to go where Naomi went [S1]."),
]


@pytest.fixture()
def tiny_seed_dir(tmp_path):
    d = tmp_path / "seed"
    write_jsonl(d / "tiny.jsonl", TINY)
    return d


def run(seed_dir, out, seed=7):
    build_dataset.main(["--seed-dir", str(seed_dir), "--no-content-db", "--out", str(out), "--seed", str(seed)])
    return list(read_jsonl(out))


def test_deterministic(tiny_seed_dir, tmp_path):
    a = run(tiny_seed_dir, tmp_path / "a.jsonl")
    b = run(tiny_seed_dir, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len(a) == len(b) > len(TINY)


def test_negatives_are_valid_refusals(tiny_seed_dir, tmp_path):
    records = run(tiny_seed_dir, tmp_path / "out.jsonl")
    by_id = {r["id"]: r for r in TINY}
    negatives = [r for r in records if "augmented:negative" in r["tags"]]
    assert negatives
    for neg in negatives:
        parent = by_id[neg["parent_id"]]
        assert validate_record(neg) == []
        assert "refusal" in neg["tags"] and neg["category"] == "missing_evidence"
        assert extract_citations(neg["answer"]) == []
        assert is_refusal(neg["answer"], len(neg["inputs"]["sources"]))
        assert neg["inputs"]["question"] == parent["inputs"]["question"]
        parent_texts = {s["text"] for s in parent["inputs"]["sources"]}
        for src in neg["inputs"]["sources"]:
            assert src["text"] not in parent_texts
            assert not src["reference"].startswith("Ruth")  # held-out books never used as pool sources
            assert src["reference"] in neg["answer"]


def test_negative_pool_excludes_same_topic():
    pool = build_pool(TINY + [seed("s-wept2", "Luke", "grief", "q", [("Luke 7:13", "Don’t cry.")], "x [S1].")])
    import random
    rng = random.Random(0)
    for _ in range(30):
        neg = make_negative(TINY[0], pool, rng)
        if neg:
            assert all(s["reference"] != "Luke 7:13" for s in neg["inputs"]["sources"])
            assert topics_of(neg) == topics_of(TINY[0])


def test_distractors_renumber_citations(tiny_seed_dir, tmp_path):
    records = run(tiny_seed_dir, tmp_path / "out.jsonl")
    by_id = {r["id"]: r for r in TINY}
    distracted = [r for r in records if "augmented:distractor" in r["tags"]]
    assert distracted
    for rec in distracted:
        parent = by_id[rec["parent_id"]]
        assert validate_record(rec) == []
        assert len(rec["inputs"]["sources"]) > len(parent["inputs"]["sources"])

        def cited_texts(r):
            return [r["inputs"]["sources"][int(c[1:]) - 1]["text"] for c in extract_citations(r["answer"])]

        assert cited_texts(rec) == cited_texts(parent)
        # Text outside the citations is unchanged.
        strip = lambda s: re.sub(r"\[S\d+\]", "", s)
        assert strip(rec["answer"]) == strip(parent["answer"])


def test_full_seed_build_is_valid_and_deterministic(tmp_path):
    seed_dir = TRAINING_DIR / "data" / "seed"
    a = run(seed_dir, tmp_path / "a.jsonl", seed=13)
    run(seed_dir, tmp_path / "b.jsonl", seed=13)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    stats = json.loads((tmp_path / "a_stats.json").read_text())
    assert stats["total"] == len(a) and stats["negatives"] > 0 and stats["distractor_variants"] > 0
    assert len({r["id"] for r in a}) == len(a)
    seeds = read_seed_dir(seed_dir)
    for task in ("memory_extraction", "summarization"):
        n_seed = sum(r.get("task") == task for r in seeds)
        assert n_seed > 0 and stats["by_task"][task] == n_seed  # passed through, not augmented


def test_teacher_drafts_need_review(tiny_seed_dir, tmp_path):
    draft = dict(TINY[0], id="teacher-1", needs_review=True, tags=["teacher_draft"])
    write_jsonl(tmp_path / "drafts.jsonl", [draft])
    base = ["--seed-dir", str(tiny_seed_dir), "--no-content-db", "--teacher-drafts", str(tmp_path / "drafts.jsonl"),
            "--negatives-per-record", "0", "--distractors-per-record", "0"]
    build_dataset.main([*base, "--out", str(tmp_path / "x.jsonl")])
    assert "teacher-1" not in {r["id"] for r in read_jsonl(tmp_path / "x.jsonl")}
    build_dataset.main([*base, "--include-unreviewed", "--out", str(tmp_path / "y.jsonl")])
    # Included, but its messages are identical to s-wept, so dedup keeps only the first copy.
    assert len(list(read_jsonl(tmp_path / "y.jsonl"))) == len(TINY)
    # A reviewed draft with a distinct question survives.
    draft2 = dict(draft, id="teacher-2", inputs=dict(draft["inputs"], question="Did Jesus cry?"), needs_review=False)
    write_jsonl(tmp_path / "drafts.jsonl", [draft2])
    build_dataset.main([*base, "--out", str(tmp_path / "z.jsonl")])
    assert "teacher-2" in {r["id"] for r in read_jsonl(tmp_path / "z.jsonl")}
