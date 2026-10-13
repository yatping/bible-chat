import json

import pytest

from bible_ft import SHARED_DIR, TRAINING_DIR
from bible_ft.aux_tasks import (
    memory_extraction_prompt,
    memory_messages,
    parse_memories,
    summarization_prompt,
    validate_memory_answer,
    validate_summary_answer,
)
from bible_ft.jsonl import read_seed_dir
from bible_ft.metrics import compute_memory_metrics, compute_summary_metrics
from bible_ft.schema import to_messages, validate_record
from bible_ft.split import leaked_questions, split_records

SEEDS = read_seed_dir(TRAINING_DIR / "data" / "seed")
MEM = [r for r in SEEDS if r.get("task") == "memory_extraction"]
SUM = [r for r in SEEDS if r.get("task") == "summarization"]
GOLDEN = json.loads((SHARED_DIR / "fixtures" / "memory_extraction_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_memory_parse_matches_golden(case):
    assert parse_memories(case["raw"]) == case["expected"]


def test_memory_prompt_shape():
    prompt = memory_extraction_prompt("  I prefer the WEB.\n")
    assert prompt.endswith("USER MESSAGE:\nI prefer the WEB.")
    assert "{{" not in prompt
    assert memory_messages("hi") == [{"role": "user", "content": memory_extraction_prompt("hi")}]


def test_summary_prompt_shape():
    long = "x" * 700
    prompt = summarization_prompt("  ", [{"role": "user", "content": " Hi "}, {"role": "assistant", "content": long}])
    assert "PREVIOUS SUMMARY:\n(none)\n" in prompt
    assert prompt.endswith("NEW MESSAGES:\nUser: Hi\nAssistant: " + "x" * 600 + "…")
    assert "{{" not in prompt
    assert "PREVIOUS SUMMARY:\nEarlier.\n" in summarization_prompt(" Earlier. ", [{"role": "user", "content": "q"}])


def test_aux_seed_counts_and_mix():
    assert 18 <= len(MEM) <= 30 and 6 <= len(SUM) <= 12
    empty = [r for r in MEM if json.loads(r["answer"])["memories"] == []]
    assert len(empty) >= 8
    types = {m["type"] for r in MEM for m in json.loads(r["answer"])["memories"]}
    assert types == {"preference", "goal", "study_interest", "progress", "note"}
    for r in MEM + SUM:
        assert validate_record(r) == [], r["id"]


def test_aux_to_messages_single_user_turn():
    for r in (MEM[0], SUM[0]):
        msgs = to_messages(r)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["content"] == r["answer"]


@pytest.mark.parametrize("answer, fragment", [
    ('Here: {"memories":[]}', "not JSON"),
    ('{"memories": []}', "compact"),
    ('{"memories":[],"extra":1}', "exactly"),
    ('{"memories":[{"type":"secret","key":null,"value":"x","importance":0.5,"confidence":1.0}]}', "not valid"),
    ('{"memories":[{"type":"goal","key":"favourite","value":"x","importance":0.5,"confidence":1.0}]}', "unknown key"),
    ('{"memories":[{"type":"goal","key":null,"value":"x","importance":0.5,"confidence":0.3}]}', "not valid"),
    ('{"memories":[{"type":"goal","key":null,"value":" x ","importance":0.5,"confidence":1.0}]}', "not valid"),
    ('{"memories":[{"type":"goal","key":null,"value":"x"}]}', "exactly type/key"),
])
def test_memory_answer_validation_rejects(answer, fragment):
    assert any(fragment in e for e in validate_memory_answer(answer)), validate_memory_answer(answer)


def test_memory_answer_too_many_items():
    item = {"type": "note", "key": None, "value": "x", "importance": 0.5, "confidence": 1.0}
    answer = json.dumps({"memories": [item] * 6}, separators=(",", ":"))
    assert any("more than 5" in e for e in validate_memory_answer(answer))


def test_summary_validation():
    assert validate_summary_answer("The user asked about grace.") == []
    assert validate_summary_answer(" ".join(["word"] * 121))
    assert validate_summary_answer("- bullet one\n- bullet two")
    assert validate_summary_answer("Summary: the user asked.")
    assert validate_summary_answer("The user asked [S1].")


def test_aux_metrics():
    gold_empty, gold_pref = '{"memories":[]}', MEM[0]["answer"]
    m = compute_memory_metrics(["nothing here", gold_pref, '{"memories":[]}'], [gold_empty, gold_pref, gold_pref])
    assert m["json_valid_rate"] == pytest.approx(2 / 3)
    assert m["empty_accuracy"] == 1.0  # unparseable output extracts nothing
    assert m["extraction_recall"] == pytest.approx(1 / 2)
    s = compute_summary_metrics(["Fine.", " ".join(["w"] * 130), ""])
    assert s["within_word_limit_rate"] == pytest.approx(2 / 3) and s["non_empty_rate"] == pytest.approx(2 / 3)


def test_split_includes_aux_randomly_without_leakage():
    rag = [{"id": f"r{i}", "book": "Ruth" if i < 3 else "Mark", "inputs": {"question": f"q{i}"}} for i in range(30)]
    splits = split_records(rag + MEM + SUM, ["Ruth"], val_fraction=0.1, seed=3)
    aux_counts = {k: sum(r.get("task") is not None for r in v) for k, v in splits.items()}
    assert sum(aux_counts.values()) == len(MEM) + len(SUM)
    assert aux_counts["test"] >= 2 and aux_counts["val"] >= 2 and aux_counts["train"] > aux_counts["test"]
    for task in ("memory_extraction", "summarization"):
        assert all(any(r.get("task") == task for r in splits[name]) for name in ("train", "val", "test"))
    assert leaked_questions(splits) == set()
