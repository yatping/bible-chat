"""Dataset record format and validation.

A record (one JSON object per line)::

    {
      "id": "seed-forgiveness-001",
      "category": "direct_scripture",
      "book": "Matthew",                    # primary book, used for the held-out test split
      "inputs": {
        "memories": ["..."],
        "summary": null,
        "sources": [{"type": "scripture", "reference": "Matthew 18:21-22", "text": "..."}],
        "recent_messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "question": "..."
      },
      "answer": "... [S1] ...",
      "tags": ["topic:forgiveness", "format:short", "refusal"]
    }

Optional keys: ``task`` (default ``rag_answer``), ``needs_review`` (teacher drafts),
``parent_id`` (augmented records), ``variants`` (seed files only: alternative memories + answer).

Auxiliary tasks (see ``aux_tasks``) use ``category == task`` and their own ``inputs``; ``book`` is
optional for them::

    {"id": ..., "task": "memory_extraction", "category": "memory_extraction",
     "inputs": {"user_message": "..."}, "answer": "{\"memories\":[...]}", "tags": [...]}
    {"id": ..., "task": "summarization", "category": "summarization",
     "inputs": {"previous_summary": null, "messages": [{"role": "user", "content": "..."}, ...]},
     "answer": "...", "tags": [...]}

``to_messages(record)`` produces exactly the runtime prompt (prompt_format) plus the gold
assistant turn, so training data == runtime prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .aux_tasks import memory_messages, summary_messages, validate_memory_answer, validate_summary_answer
from .citations import extract_citations, normalise_citations, validate_citations
from .metrics import FORMAT_TAGS, has_refusal_phrase
from .prompt_format import Message, build_messages_from_inputs

CATEGORIES = frozenset(
    {
        "direct_scripture",
        "explanation",
        "comparison",
        "historical_context",
        "theme_search",
        "source_grounding",
        "missing_evidence",
        "conflicting_interpretations",
        "formatting",
        "distractor_citation",
    }
)
RAG_TASK = "rag_answer"
AUX_TASKS = frozenset({"memory_extraction", "summarization"})
TASKS = frozenset({RAG_TASK}) | AUX_TASKS
SOURCE_TYPES = frozenset({"scripture", "commentary", "dictionary", "cross_reference"})
ROLES = frozenset({"user", "assistant"})
REFUSAL_TAG = "refusal"

_REQUIRED = {"id", "category", "book", "inputs", "answer", "tags"}
_OPTIONAL = {"task", "needs_review", "parent_id", "variants"}
_AUX_REQUIRED = {"id", "task", "category", "inputs", "answer", "tags"}
_AUX_OPTIONAL = {"book", "needs_review", "parent_id"}
_AUX_INPUT_KEYS = {"memory_extraction": {"user_message"}, "summarization": {"previous_summary", "messages"}}
_INPUT_KEYS = {"memories", "summary", "sources", "recent_messages", "question"}
_VARIANT_KEYS = {"memories", "answer", "tags", "summary"}
_CONTROL_TOKEN_RE = re.compile(r"<(?:start_of_turn|end_of_turn|bos|eos|\|turn|turn\||\|think)")
_MEMORY_CITE_RE = re.compile(r"\[M\d+\]")


class ValidationError(ValueError):
    pass


def task_of(record: Mapping[str, Any]) -> str:
    return str(record.get("task", RAG_TASK))


def is_rag(record: Mapping[str, Any]) -> bool:
    return task_of(record) == RAG_TASK


def is_refusal_record(record: Mapping[str, Any]) -> bool:
    return is_rag(record) and REFUSAL_TAG in record.get("tags", [])


def prompt_messages(record: Mapping[str, Any]) -> list[Message]:
    """Runtime prompt messages only (for generation / evaluation), per task."""
    task = task_of(record)
    inputs = record["inputs"]
    if task == "memory_extraction":
        return memory_messages(inputs["user_message"])
    if task == "summarization":
        return summary_messages(inputs["previous_summary"], inputs["messages"])
    return build_messages_from_inputs(inputs)


def to_messages(record: Mapping[str, Any]) -> list[Message]:
    """Runtime prompt messages + the gold assistant answer as the final turn."""
    messages = prompt_messages(record)
    messages.append({"role": "assistant", "content": record["answer"]})
    return messages


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_record(record: Mapping[str, Any], *, allow_variants: bool = False) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["record is not an object"]

    if record.get("task", RAG_TASK) not in TASKS:
        return [f"unknown task {record.get('task')!r}"]
    if not is_rag(record):
        return _validate_aux(record)

    keys = set(record)
    if missing := _REQUIRED - keys:
        errors.append(f"missing keys: {sorted(missing)}")
    allowed = _REQUIRED | _OPTIONAL if allow_variants else _REQUIRED | (_OPTIONAL - {"variants"})
    if extra := keys - allowed:
        errors.append(f"unknown keys: {sorted(extra)}")
    if errors:
        return errors

    if not _nonempty_str(record["id"]):
        errors.append("id must be a non-empty string")
    if record["category"] not in CATEGORIES:
        errors.append(f"unknown category {record['category']!r}")
    if not _nonempty_str(record["book"]):
        errors.append("book must be a non-empty string")
    tags = record["tags"]
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        errors.append("tags must be a list of strings")
        return errors
    if "needs_review" in record and not isinstance(record["needs_review"], bool):
        errors.append("needs_review must be a bool")

    errors.extend(_validate_inputs(record["inputs"]))
    if errors:
        return errors

    errors.extend(_validate_answer(record["answer"], record["inputs"], tags, record["category"]))

    if allow_variants and "variants" in record:
        variants = record["variants"]
        if not isinstance(variants, list) or not variants:
            errors.append("variants must be a non-empty list")
        else:
            for i, variant in enumerate(variants):
                for err in validate_record(expand_variant(record, variant, i)):
                    errors.append(f"variant {i}: {err}")
    return errors


def _validate_messages(messages: Any, name: str) -> list[str]:
    if not isinstance(messages, list):
        return [f"{name} must be a list"]
    errors: list[str] = []
    prev_role = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, Mapping) or msg.get("role") not in ROLES or not _nonempty_str(msg.get("content")):
            errors.append(f"{name}[{i}] must have role user/assistant and content")
            continue
        if msg["role"] == prev_role:
            errors.append(f"{name}[{i}] repeats role {prev_role!r}")
        prev_role = msg["role"]
    return errors


def _validate_aux(record: Mapping[str, Any]) -> list[str]:
    task = record["task"]
    keys = set(record)
    errors: list[str] = []
    if missing := _AUX_REQUIRED - keys:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra := keys - _AUX_REQUIRED - _AUX_OPTIONAL:
        errors.append(f"unknown keys: {sorted(extra)}")
    if errors:
        return errors
    if not _nonempty_str(record["id"]):
        errors.append("id must be a non-empty string")
    if record["category"] != task:
        errors.append("auxiliary records must have category == task")
    if not isinstance(record["tags"], list) or not all(isinstance(t, str) for t in record["tags"]):
        errors.append("tags must be a list of strings")
    inputs = record["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != _AUX_INPUT_KEYS[task]:
        return errors + [f"inputs must have exactly {sorted(_AUX_INPUT_KEYS[task])}"]
    answer = record["answer"]
    if not _nonempty_str(answer) or answer != answer.strip():
        return errors + ["answer must be a non-empty, trimmed string"]

    if task == "memory_extraction":
        if not _nonempty_str(inputs["user_message"]):
            errors.append("user_message must be a non-empty string")
        errors.extend(validate_memory_answer(answer))
    else:
        prev = inputs["previous_summary"]
        if prev is not None and not isinstance(prev, str):
            errors.append("previous_summary must be a string or null")
        if not inputs["messages"]:
            errors.append("messages must not be empty")
        errors.extend(_validate_messages(inputs["messages"], "messages"))
        errors.extend(validate_summary_answer(answer))
    return errors


def _validate_inputs(inputs: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(inputs, Mapping):
        return ["inputs must be an object"]
    if missing := _INPUT_KEYS - set(inputs):
        errors.append(f"inputs missing keys: {sorted(missing)}")
    if extra := set(inputs) - _INPUT_KEYS:
        errors.append(f"inputs unknown keys: {sorted(extra)}")
    if errors:
        return errors

    memories = inputs["memories"]
    if not isinstance(memories, list) or not all(_nonempty_str(m) for m in memories):
        errors.append("memories must be a list of non-empty strings")
    summary = inputs["summary"]
    if summary is not None and not isinstance(summary, str):
        errors.append("summary must be a string or null")
    if not _nonempty_str(inputs["question"]):
        errors.append("question must be a non-empty string")

    sources = inputs["sources"]
    if not isinstance(sources, list):
        errors.append("sources must be a list")
    else:
        for i, src in enumerate(sources, start=1):
            if not isinstance(src, Mapping) or set(src) != {"type", "reference", "text"}:
                errors.append(f"S{i} must have exactly type/reference/text")
                continue
            if src["type"] not in SOURCE_TYPES:
                errors.append(f"S{i} has unknown type {src['type']!r}")
            if not _nonempty_str(src["reference"]) or not _nonempty_str(src["text"]):
                errors.append(f"S{i} reference/text must be non-empty")

    recent = inputs["recent_messages"]
    errors.extend(_validate_messages(recent, "recent_messages"))
    if isinstance(recent, list) and recent and isinstance(recent[-1], Mapping) and recent[-1].get("role") != "assistant":
        errors.append("recent_messages must end with an assistant turn")
    return errors


def _validate_answer(answer: Any, inputs: Mapping[str, Any], tags: list[str], category: str) -> list[str]:
    errors: list[str] = []
    if not _nonempty_str(answer):
        return ["answer must be a non-empty string"]
    if answer != answer.strip():
        errors.append("answer has leading/trailing whitespace")
    if _CONTROL_TOKEN_RE.search(answer):
        errors.append("answer contains chat control tokens")
    if _MEMORY_CITE_RE.search(answer):
        errors.append("answer cites a memory id ([M#]); only [S#] may be cited")
    if normalise_citations(answer) != answer:
        errors.append("answer uses grouped/ranged citations; write [S1][S2]")

    n_sources = len(inputs["sources"])
    result = validate_citations(answer, n_sources)
    if result.removed:
        errors.append(f"answer cites ids not in SOURCES: {result.removed}")

    refusal = REFUSAL_TAG in tags
    if refusal:
        if extract_citations(answer):
            errors.append("refusal answer must not cite anything")
        if not has_refusal_phrase(answer):
            errors.append("refusal answer is not detected by metrics.has_refusal_phrase")
    else:
        if category == "missing_evidence":
            errors.append("missing_evidence records must be tagged 'refusal'")
        if n_sources == 0:
            errors.append("answerable record has no sources")
        elif not result.cited:
            errors.append("answerable record must cite at least one source")

    fmt_tags = [t for t in tags if t.startswith("format:")]
    for t in fmt_tags:
        if t.split(":", 1)[1] not in FORMAT_TAGS:
            errors.append(f"unknown format tag {t!r}")
    return errors


def expand_variant(record: Mapping[str, Any], variant: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Materialise a seed ``variants`` entry into a standalone record."""
    if extra := set(variant) - _VARIANT_KEYS:
        raise ValidationError(f"variant {index} has unknown keys {sorted(extra)}")
    inputs = dict(record["inputs"])
    if "memories" in variant:
        inputs["memories"] = list(variant["memories"])
    if "summary" in variant:
        inputs["summary"] = variant["summary"]
    base_tags = [t for t in record["tags"] if not t.startswith("format:")]
    return {
        "id": f"{record['id']}__v{index + 1}",
        "category": record["category"],
        "book": record["book"],
        "inputs": inputs,
        "answer": variant["answer"],
        "tags": sorted(set(base_tags) | set(variant.get("tags", [])) | {"augmented:variant"}),
        "parent_id": record["id"],
    }


def validate_dataset(records: Iterable[Mapping[str, Any]], *, allow_variants: bool = False) -> dict[str, list[str]]:
    """Validate many records; also flags duplicate ids. Returns {id: errors} for failures."""
    failures: dict[str, list[str]] = {}
    seen: set[str] = set()
    for i, record in enumerate(records):
        rid = str(record.get("id", f"<line {i + 1}>")) if isinstance(record, Mapping) else f"<line {i + 1}>"
        errs = validate_record(record, allow_variants=allow_variants)
        if rid in seen:
            errs.append("duplicate id")
        seen.add(rid)
        if errs:
            failures[rid] = errs
    return failures
