import json

import pytest

from bible_ft import SHARED_DIR
from bible_ft.prompt_format import build_messages_from_inputs, load_system_prompt

GOLDEN = json.loads((SHARED_DIR / "fixtures" / "prompt_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_prompt_golden(case):
    assert build_messages_from_inputs(case["input"]) == case["expected_messages"]


def test_system_prompt_has_no_trailing_newline():
    prompt = load_system_prompt()
    assert prompt.startswith("You are an offline Bible study assistant.")
    assert not prompt.endswith("\n")
