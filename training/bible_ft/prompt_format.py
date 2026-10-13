"""Build chat messages exactly as the iOS app does (shared/CONTRACTS.md §1).

The model is fine-tuned on the same messages the app sends at runtime, so this module must
reproduce ``shared/fixtures/prompt_golden.json`` byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import SHARED_DIR

SYSTEM_PROMPT_PATH = SHARED_DIR / "system_prompt.txt"
NO_SOURCES_LINE = "(no sources retrieved)"

Message = dict[str, str]


@lru_cache(maxsize=4)
def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Return the shared system prompt with its trailing newline(s) stripped."""
    return path.read_text(encoding="utf-8").rstrip("\n")


def format_source(index: int, source: Mapping[str, Any]) -> str:
    """Render one source as ``[S<index>] type: ... | reference: ...\\n<text>``."""
    return (
        f"[S{index}] type: {str(source['type']).strip()} | "
        f"reference: {str(source['reference']).strip()}\n"
        f"{str(source['text']).strip()}"
    )


def build_context_block(
    memories: Sequence[str],
    summary: str | None,
    sources: Sequence[Mapping[str, Any]],
    question: str,
) -> str:
    """Build the final user message: MEMORY, CONVERSATION SUMMARY, SOURCES, QUESTION."""
    sections: list[str] = []

    # Memories are numbered in the given order; the app never passes empty ones.
    cleaned_memories = [m.strip() for m in memories]
    if cleaned_memories:
        lines = [f"[M{i}] {m}" for i, m in enumerate(cleaned_memories, start=1)]
        sections.append("MEMORY\n" + "\n".join(lines))

    if summary is not None and summary.strip():
        sections.append("CONVERSATION SUMMARY\n" + summary.strip())

    if sources:
        body = "\n\n".join(format_source(i, s) for i, s in enumerate(sources, start=1))
    else:
        body = NO_SOURCES_LINE
    sections.append("SOURCES\n" + body)

    sections.append("QUESTION\n" + question.strip())
    return "\n\n".join(sections)


def build_messages(
    memories: Sequence[str],
    summary: str | None,
    sources: Sequence[Mapping[str, Any]],
    recent_messages: Sequence[Mapping[str, str]],
    question: str,
    system_prompt: str | None = None,
) -> list[Message]:
    """Return ``[system, *recent_messages, user(context block)]`` per CONTRACTS §1.

    Recent messages are replayed with their raw content (historical user turns are the raw
    question, historical assistant turns the final validated answer).
    """
    messages: list[Message] = [
        {"role": "system", "content": system_prompt if system_prompt is not None else load_system_prompt()}
    ]
    for msg in recent_messages:
        messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
    messages.append(
        {"role": "user", "content": build_context_block(memories, summary, sources, question)}
    )
    return messages


def build_messages_from_inputs(inputs: Mapping[str, Any]) -> list[Message]:
    """Convenience wrapper taking a dataset ``inputs`` / golden ``input`` mapping."""
    return build_messages(
        memories=inputs.get("memories") or [],
        summary=inputs.get("summary"),
        sources=inputs.get("sources") or [],
        recent_messages=inputs.get("recent_messages") or [],
        question=inputs["question"],
    )
