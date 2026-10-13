"""Auxiliary tasks the same on-device model performs: memory extraction and summarisation.

Message shape (mirrors ios/BibleAICore MemoryExtractor / ConversationSummarizer): a SINGLE user turn
containing the filled template, no system message.

- memory_extraction: ``shared/memory_extraction_prompt.txt`` with trailing newlines stripped and
  ``{{user_message}}`` replaced by the trimmed user message.
- summarization: ``shared/summarization_prompt.txt`` with trailing newlines stripped,
  ``{{previous_summary}}`` = trimmed previous summary or ``(none)`` if empty, and ``{{messages}}`` =
  one line per message, ``User: <content>`` / ``Assistant: <content>``, content trimmed and capped at
  600 characters (+ ``…``), joined with ``\\n``.

Memory output parsing/validation mirrors CONTRACTS §4 (and MemoryExtractor.parse on iOS).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import SHARED_DIR

MEMORY_PROMPT_PATH = SHARED_DIR / "memory_extraction_prompt.txt"
SUMMARY_PROMPT_PATH = SHARED_DIR / "summarization_prompt.txt"

MEMORY_TYPES = frozenset({"preference", "goal", "study_interest", "progress", "note"})
MEMORY_KEYS = frozenset({
    "bible_translation", "answer_style", "answer_length", "explanation_level",
    "theological_tradition", "current_study", "reading_plan",
})
MAX_MEMORY_ITEMS = 5
MAX_VALUE_CHARS = 300
MIN_CONFIDENCE = 0.5
MAX_SUMMARY_WORDS = 120
MAX_CHARS_PER_MESSAGE = 600

Message = dict[str, str]


@lru_cache(maxsize=4)
def _template(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def memory_extraction_prompt(user_message: str) -> str:
    return _template(MEMORY_PROMPT_PATH).replace("{{user_message}}", user_message.strip())


def summarization_prompt(previous_summary: str | None, messages: Sequence[Mapping[str, str]]) -> str:
    previous = (previous_summary or "").strip()
    lines = []
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].strip()
        if len(content) > MAX_CHARS_PER_MESSAGE:
            content = content[:MAX_CHARS_PER_MESSAGE] + "…"
        lines.append(f"{speaker}: {content}")
    return (
        _template(SUMMARY_PROMPT_PATH)
        .replace("{{previous_summary}}", previous or "(none)")
        .replace("{{messages}}", "\n".join(lines))
    )


def memory_messages(user_message: str) -> list[Message]:
    return [{"role": "user", "content": memory_extraction_prompt(user_message)}]


def summary_messages(previous_summary: str | None, messages: Sequence[Mapping[str, str]]) -> list[Message]:
    return [{"role": "user", "content": summarization_prompt(previous_summary, messages)}]


# ---- memory output parsing (CONTRACTS §4) ----

def first_json_object(text: str) -> str | None:
    """First balanced ``{...}`` substring, honouring JSON string literals and escapes."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.5


def parse_memory_object(raw: str) -> dict[str, Any] | None:
    """The first balanced JSON object with a ``memories`` list, else None (= unparseable output)."""
    candidate = first_json_object(raw)
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("memories"), list):
        return None
    return obj


def validate_memory_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or not isinstance(item.get("type"), str):
        return None
    mtype = item["type"].strip().lower()
    if mtype not in MEMORY_TYPES:
        return None
    key = item.get("key")
    key = key.strip().lower() if isinstance(key, str) else None
    if key not in MEMORY_KEYS:
        key = None
    value = item.get("value")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_VALUE_CHARS:
        return None
    importance = _number(item.get("importance"))
    confidence = _number(item.get("confidence"))
    importance = _clamp(importance) if importance is not None else 0.5
    confidence = _clamp(confidence) if confidence is not None else 1.0
    if confidence < MIN_CONFIDENCE:
        return None
    return {"type": mtype, "key": key, "value": value, "importance": importance, "confidence": confidence}


def parse_memories(raw: str) -> list[dict[str, Any]]:
    """App-side parse: valid items only, at most 5 (matches memory_extraction_golden.json)."""
    obj = parse_memory_object(raw)
    if obj is None:
        return []
    out: list[dict[str, Any]] = []
    for item in obj["memories"]:
        if (valid := validate_memory_item(item)) is not None:
            out.append(valid)
            if len(out) == MAX_MEMORY_ITEMS:
                break
    return out


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def validate_memory_answer(answer: str) -> list[str]:
    """Gold answers must be strict, canonical, fully-valid JSON (no prose, nothing the app would drop)."""
    try:
        obj = json.loads(answer)
    except json.JSONDecodeError as exc:
        return [f"answer is not JSON: {exc}"]
    if not isinstance(obj, dict) or set(obj) != {"memories"} or not isinstance(obj["memories"], list):
        return ['answer must be exactly {"memories":[...]}']
    errors: list[str] = []
    items = obj["memories"]
    if len(items) > MAX_MEMORY_ITEMS:
        errors.append(f"more than {MAX_MEMORY_ITEMS} memories")
    for i, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"type", "key", "value", "importance", "confidence"}:
            errors.append(f"memory {i} must have exactly type/key/value/importance/confidence")
            continue
        if validate_memory_item(item) != item:
            errors.append(f"memory {i} is not valid per CONTRACTS §4 (type/key/value/ranges)")
        if item["key"] is not None and item["key"] not in MEMORY_KEYS:
            errors.append(f"memory {i} has unknown key {item['key']!r}")
    if answer != compact_json(obj):
        errors.append("answer must be compact JSON (separators ',' ':', no extra whitespace)")
    return errors


def validate_summary_answer(answer: str) -> list[str]:
    errors: list[str] = []
    words = len(re.findall(r"\S+", answer))
    if words > MAX_SUMMARY_WORDS:
        errors.append(f"summary has {words} words (> {MAX_SUMMARY_WORDS})")
    if re.search(r"\[[SM]\d+\]", answer):
        errors.append("summary must not contain [S#]/[M#] ids")
    if re.search(r"^\s*(?:[-*•]|\d+[.)])\s", answer, re.MULTILINE) or "\n" in answer:
        errors.append("summary must be a single paragraph of plain prose")
    if answer.lower().startswith(("summary:", "here is", "here's")):
        errors.append("summary must be returned without a preamble")
    return errors
