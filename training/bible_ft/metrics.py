"""Pure evaluation metrics for source-grounded answering.

Refusal detection heuristic
---------------------------
An answer counts as a *refusal* when BOTH hold:

1. it contains one of ``REFUSAL_PATTERNS`` (case-insensitive), e.g. "the supplied sources do
   not cover", "not enough information", "I can't answer ... from the sources"; and
2. it contains no valid ``[S#]`` citation (contract §1: a refusal cites nothing).

Rule 2 keeps partially-grounded answers that add a caveat ("the sources don't say whether
...") from being counted as refusals. The flip side: a model that refuses *and* cites is
scored as answering, which is the behaviour we want to penalise anyway.

Format checks (driven by ``format:*`` tags on the gold record)
--------------------------------------------------------------
- ``format:short``        <= 90 words
- ``format:bullets``      >= 2 lines starting with ``-``, ``*``, ``•`` or ``1.``
- ``format:study_notes``  >= 2 bullet/numbered lines or >= 2 heading-like lines (``#`` or ``Label:``)
- ``format:detailed``     >= 120 words
- ``format:one_sentence`` exactly one sentence-ending punctuation run (<= 45 words)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .citations import extract_citations, validate_citations

REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\b(?:sources?|passages?|context)\b[^.]{0,60}\b(?:do(?:es)?\s*n[o']t|don't|doesn't)\s+"
    r"(?:cover|contain|address|mention|say|answer|discuss|include|describe|tell|provide|support|speak)",
    r"\bnot\s+(?:covered|addressed|mentioned|answered)\s+(?:by|in)\s+the\s+(?:supplied|provided|retrieved|available)?\s*sources?",
    r"\b(?:not|isn't|is\s+not)\s+(?:enough|sufficient)\s+(?:information|evidence|context)",
    r"\binsufficient\s+(?:information|evidence|context|sources)",
    r"\bno\s+sources?\s+(?:were\s+)?(?:retrieved|supplied|provided|available)",
    r"\b(?:can(?:no|')t|cannot|can\s+not|unable\s+to)\s+(?:answer|say|confirm|tell|determine)\b[^.]{0,80}"
    r"\b(?:from|with|based\s+on|using)\s+(?:the|these|this)\b",
    r"\bnothing\s+in\s+the\s+(?:supplied|provided|retrieved|available)?\s*sources?",
)
_REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+\S.*|\*\*[^*\n]+\*\*:?\s*$|[A-Z][A-Za-z /]{1,40}:\s*$)", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9’']+")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"”’)\]]*)(?=\s|$)")

FORMAT_TAGS = ("short", "bullets", "study_notes", "detailed", "one_sentence")


def has_refusal_phrase(text: str) -> bool:
    return _REFUSAL_RE.search(text) is not None


def is_refusal(text: str, source_count: int) -> bool:
    """Heuristic refusal detection (see module docstring)."""
    return has_refusal_phrase(text) and not validate_citations(text, source_count).cited


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(re.sub(r"\[S\d+\]", " ", text)))


def format_tag(tags: Iterable[str]) -> str | None:
    for tag in tags:
        if tag.startswith("format:") and tag.split(":", 1)[1] in FORMAT_TAGS:
            return tag.split(":", 1)[1]
    return None


def check_format(text: str, fmt: str) -> bool:
    words = word_count(text)
    bullets = len(_BULLET_RE.findall(text))
    if fmt == "short":
        return words <= 90
    if fmt == "bullets":
        return bullets >= 2
    if fmt == "study_notes":
        return bullets >= 2 or len(_HEADING_RE.findall(text)) >= 2
    if fmt == "detailed":
        return words >= 120
    if fmt == "one_sentence":
        stripped = re.sub(r"\s*\[S\d+\]", "", text).strip()
        return words <= 45 and len(_SENTENCE_END_RE.findall(stripped)) == 1
    raise ValueError(f"unknown format {fmt!r}")


@dataclass(frozen=True)
class EvalItem:
    """One prediction plus what the gold record says about it."""

    prediction: str
    source_count: int
    expect_refusal: bool
    gold_cited: tuple[str, ...] = ()
    format: str | None = None


@dataclass(frozen=True)
class Metrics:
    n: int
    n_answerable: int
    n_refusal_expected: int
    citation_validity_rate: float | None
    unsupported_citation_rate: float | None
    answers_with_invalid_citation_rate: float | None
    correct_refusal_rate: float | None
    false_refusal_rate: float | None
    citation_coverage: float | None
    format_adherence: float | None
    n_format_items: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def compute_metrics(items: Sequence[EvalItem]) -> Metrics:
    """Aggregate metrics.

    - citation_validity_rate: in-range citations / all citations emitted.
    - unsupported_citation_rate: in-range citations of a source the gold answer does not cite
      (distractors; any citation on a refusal item) / all in-range citations.
    - answers_with_invalid_citation_rate: answers with >= 1 out-of-range id / all answers.
    - correct_refusal_rate: refusals / items whose gold answer is a refusal.
    - false_refusal_rate: refusals / answerable items.
    - citation_coverage: answerable items whose prediction cites >= 1 valid source / answerable items.
    - format_adherence: items passing their ``format:*`` check / items carrying such a tag.
    """
    total_cites = valid_cites = unsupported = with_invalid = 0
    refusal_expected = correct_refusals = 0
    answerable = false_refusals = covered = 0
    fmt_total = fmt_ok = 0

    for item in items:
        emitted = extract_citations(item.prediction)
        result = validate_citations(item.prediction, item.source_count)
        total_cites += len(emitted)
        valid_cites += len(result.cited)
        if result.removed:
            with_invalid += 1
        gold = set(item.gold_cited)
        unsupported += sum(1 for c in result.cited if c not in gold)

        refused = is_refusal(item.prediction, item.source_count)
        if item.expect_refusal:
            refusal_expected += 1
            correct_refusals += refused
        else:
            answerable += 1
            false_refusals += refused
            covered += bool(result.cited)

        if item.format:
            fmt_total += 1
            fmt_ok += check_format(item.prediction, item.format)

    return Metrics(
        n=len(items),
        n_answerable=answerable,
        n_refusal_expected=refusal_expected,
        citation_validity_rate=_rate(valid_cites, total_cites),
        unsupported_citation_rate=_rate(unsupported, valid_cites),
        answers_with_invalid_citation_rate=_rate(with_invalid, len(items)),
        correct_refusal_rate=_rate(correct_refusals, refusal_expected),
        false_refusal_rate=_rate(false_refusals, answerable),
        citation_coverage=_rate(covered, answerable),
        format_adherence=_rate(fmt_ok, fmt_total),
        n_format_items=fmt_total,
    )


# ---- auxiliary tasks ----

def _type_keys(memories: Sequence[dict]) -> list[tuple[str, str]]:
    return sorted((m["type"], m["key"] or "") for m in memories)


def compute_memory_metrics(predictions: Sequence[str], golds: Sequence[str]) -> dict[str, float | int | None]:
    """Memory-extraction metrics (raw model output vs gold JSON answer).

    - json_valid_rate: the app can parse a ``{"memories": [...]}`` object (first balanced ``{...}``).
    - strict_json_rate: the whole output is that JSON object (no prose / code fences).
    - type_key_exact_match: multiset of (type, key) of app-validated items equals the gold's.
    - empty_accuracy: on items whose gold is ``{"memories":[]}``, the app extracts nothing.
    - extraction_recall: on items with gold memories, (type, key) multiset matches.
    """
    import json

    from .aux_tasks import parse_memories, parse_memory_object

    valid = strict = exact = empty_ok = empty_n = recall_ok = recall_n = 0
    for raw, gold_raw in zip(predictions, golds, strict=True):
        valid += parse_memory_object(raw) is not None
        try:
            obj = json.loads(raw.strip())
            strict += isinstance(obj, dict) and isinstance(obj.get("memories"), list)
        except json.JSONDecodeError:
            pass
        pred, gold = _type_keys(parse_memories(raw)), _type_keys(parse_memories(gold_raw))
        match = pred == gold
        exact += match
        if gold:
            recall_n += 1
            recall_ok += match
        else:
            empty_n += 1
            empty_ok += not pred
    n = len(predictions)
    return {
        "n": n,
        "json_valid_rate": _rate(valid, n),
        "strict_json_rate": _rate(strict, n),
        "type_key_exact_match": _rate(exact, n),
        "empty_accuracy": _rate(empty_ok, empty_n),
        "extraction_recall": _rate(recall_ok, recall_n),
    }


def compute_summary_metrics(predictions: Sequence[str]) -> dict[str, float | int | None]:
    """Summarisation metrics: non-empty, <= 120 words, and passes all summary format rules."""
    from .aux_tasks import MAX_SUMMARY_WORDS, validate_summary_answer

    texts = [p.strip() for p in predictions]
    n = len(texts)
    return {
        "n": n,
        "non_empty_rate": _rate(sum(bool(t) for t in texts), n),
        "within_word_limit_rate": _rate(sum(len(t.split()) <= MAX_SUMMARY_WORDS for t in texts), n),
        "format_ok_rate": _rate(sum(bool(t) and not validate_summary_answer(t) for t in texts), n),
    }
