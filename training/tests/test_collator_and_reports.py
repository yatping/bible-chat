"""Torch-free checks of the training collator's masking and the report helpers."""

from compare_reports import render_table
from evaluate import metrics_from_predictions, render_report
from train_qlora import AssistantOnlyCollator


class FakeTokenizer:
    """Character-level tokenizer: token id == ord(char)."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "<bos>"
        for m in messages:
            out += f"<turn>{m['role']}\n{m['content']}<end>\n"
        if add_generation_prompt:
            out += "<turn>assistant\n"
        return out


def test_collator_masks_everything_but_final_answer():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer [S1]"},
        {"role": "user", "content": "SOURCES\n...\n\nQUESTION\nq"},
        {"role": "assistant", "content": "Final answer [S1]."},
    ]
    ids, labels = AssistantOnlyCollator(FakeProcessor(), max_length=10_000).encode(messages)
    learned = "".join(chr(t) for t, l in zip(ids, labels) if l != -100)
    assert learned == "Final answer [S1].<end>\n"
    assert len(ids) == len(labels)


def test_collator_truncates():
    messages = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y"}]
    ids, labels = AssistantOnlyCollator(FakeProcessor(), max_length=20).encode(messages)
    assert len(ids) == len(labels) == 20


def test_report_rendering():
    preds = [
        {"id": "a", "category": "direct_scripture", "expect_refusal": False, "gold_cited": ["S1"], "format": None,
         "source_count": 1, "raw_output": "Jesus wept [S1].", "removed": [], "predicted_refusal": False},
        {"id": "b", "category": "missing_evidence", "expect_refusal": True, "gold_cited": [], "format": None,
         "source_count": 1, "raw_output": "It was a whale [S3].", "removed": ["S3"], "predicted_refusal": False},
    ]
    metrics = metrics_from_predictions(preds)
    assert metrics["overall"]["correct_refusal_rate"] == 0.0
    report = render_report("base", metrics, preds)
    assert "missed refusal" in report and "invalid citations" in report
    table = render_table([("base", metrics), ("finetuned", metrics)])
    assert table.splitlines()[0] == "| Metric | base | finetuned |"
    assert "Memory:" not in table  # no memory items -> rows omitted


def test_aux_predictions_in_metrics_and_table():
    preds = [
        {"id": "m1", "task": "memory_extraction", "category": "memory_extraction",
         "gold": '{"memories":[]}', "raw_output": '{"memories":[]}'},
        {"id": "m2", "task": "memory_extraction", "category": "memory_extraction",
         "gold": '{"memories":[{"type":"goal","key":"reading_plan","value":"Read the Bible.","importance":0.8,"confidence":1.0}]}',
         "raw_output": "Sure: {\"memories\":[{\"type\":\"goal\",\"key\":\"reading_plan\",\"value\":\"x\"}]}"},
        {"id": "s1", "task": "summarization", "category": "summarization", "gold": "x", "raw_output": "A short summary."},
    ]
    metrics = metrics_from_predictions(preds)
    mem = metrics["memory_extraction"]
    assert mem["n"] == 2 and mem["json_valid_rate"] == 1.0 and mem["strict_json_rate"] == 0.5
    assert mem["type_key_exact_match"] == 1.0 and mem["empty_accuracy"] == 1.0
    assert metrics["summarization"]["format_ok_rate"] == 1.0
    assert metrics["overall"]["n"] == 0
    table = render_table([("ft", metrics)])
    assert "Memory: JSON valid" in table and "Summary: <= 120 words" in table
    assert "missed refusal" not in render_report("ft", metrics, preds)
