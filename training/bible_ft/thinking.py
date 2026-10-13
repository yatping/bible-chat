"""Gemma 4 thought-channel handling (mirrors ios ``ThoughtChannelFilter``).

Base Gemma 4 sometimes opens ``<|channel>thought … <channel|>`` before answering even with thinking disabled.
The app strips these blocks before display, storage and memory-JSON parsing; evaluation does the same so models
are scored on what the user would see, and reports how often reasoning leaked.
"""

from __future__ import annotations

import re

OPEN = "<|channel>"
CLOSE = "<channel|>"
_BLOCK = re.compile(re.escape(OPEN) + r".*?(?:" + re.escape(CLOSE) + r"|\Z)", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, bool]:
    """Return ``(visible_text, had_thinking)``; an unterminated block is dropped to the end of the text."""
    visible, count = _BLOCK.subn("", text)
    return visible.strip(), count > 0
