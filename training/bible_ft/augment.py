"""Deterministic dataset augmentation (used by scripts/build_dataset.py).

- ``make_negative``: pair a question with unrelated sources -> refusal answer.
- ``add_distractors``: insert irrelevant sources and renumber the answer's citations.
- ``expand_variants``: materialise seed memory/format variants.
- ``dedup``: drop records whose rendered messages are identical.

All randomness flows through an explicit ``random.Random`` so output is reproducible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .citations import renumber_citations
from .schema import REFUSAL_TAG, expand_variant, is_rag, is_refusal_record, to_messages

Record = dict[str, Any]

REFUSAL_TEMPLATES: tuple[str, ...] = (
    "The supplied sources do not cover this question. They are about {refs}, so I can't answer it from them without guessing.",
    "I can't answer that from the provided sources. The passages supplied ({refs}) do not address it.",
    "The retrieved sources do not contain enough information to answer this. They discuss {refs} instead. You could try rephrasing the question or naming a specific passage.",
    "None of the supplied sources speak to this question; they cover {refs}. Rather than guess, I'll say that the sources do not cover it.",
    "There is not enough information in the supplied sources to answer this. The sources provided ({refs}) deal with a different subject.",
)
NO_SOURCE_TEMPLATES: tuple[str, ...] = (
    "No sources were retrieved for this question, so I can't answer it from the sources. Try naming a book, chapter or verse.",
    "There is not enough information to answer this: no sources were supplied. If you can point me to a passage, I can look at it with you.",
    "The supplied sources do not cover this, because no sources were retrieved. I'd rather not guess at biblical details without them.",
)

_WORD_RE = re.compile(r"[a-z]+")
_STOP = frozenset(
    "about after again against also among because been before being between both could does doing during each "
    "from have having into just more most other over said same should some such than that their them then there "
    "these they this those through under until very what when where which while whom whose will with would your "
    "jesus bible god lord says say tell mean meaning teach teaches taught".split()
)
_BOOK_RE = re.compile(r"^((?:[1-3]\s)?[A-Z][a-z]+(?:\s(?:of\s)?[A-Z][a-z]+)?)")


@dataclass(frozen=True)
class PoolSource:
    """A candidate distractor / negative source with its book and topic labels."""

    source: Mapping[str, str]
    book: str
    topics: frozenset[str]


def topics_of(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(t for t in record.get("tags", []) if t.startswith("topic:"))


def book_of_reference(reference: str) -> str:
    m = _BOOK_RE.match(reference.strip())
    return m.group(1) if m else reference.strip()


def content_words(text: str) -> set[str]:
    """Crudely stemmed (first 6 letters) content words, so "pharisee" ~ "pharisees"."""
    return {w[:6] for w in _WORD_RE.findall(text.lower()) if len(w) >= 5 and w not in _STOP}


def build_pool(records: Iterable[Mapping[str, Any]], exclude_books: Iterable[str] = ()) -> list[PoolSource]:
    """Unique sources from records, labelled with the record's topics."""
    excluded = set(exclude_books)
    pool: dict[tuple[str, str], PoolSource] = {}
    for record in records:
        if not is_rag(record):
            continue
        for src in record["inputs"]["sources"]:
            book = book_of_reference(src["reference"])
            if book in excluded or record["book"] in excluded:
                continue
            key = (src["reference"], src["text"])
            if key not in pool:
                pool[key] = PoolSource(source=dict(src), book=book, topics=topics_of(record))
    return [pool[k] for k in sorted(pool)]


def is_unrelated(record: Mapping[str, Any], candidate: PoolSource) -> bool:
    """Conservative relevance filter: different topic, different book, little word overlap."""
    if candidate.topics & topics_of(record):
        return False
    inputs = record["inputs"]
    used_books = {record["book"]} | {book_of_reference(s["reference"]) for s in inputs["sources"]}
    if candidate.book in used_books:
        return False
    if any(candidate.source["text"] == s["text"] for s in inputs["sources"]):
        return False
    question_words = content_words(inputs["question"])
    overlap = question_words & content_words(candidate.source["text"] + " " + candidate.source["reference"])
    return len(overlap) == 0


def _pick_unrelated(record: Mapping[str, Any], pool: Sequence[PoolSource], k: int, rng: random.Random) -> list[PoolSource]:
    candidates = [c for c in pool if is_unrelated(record, c)]
    if len(candidates) < k:
        return []
    return rng.sample(candidates, k)


def _short_hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:8]


def _format_refs(refs: Sequence[str]) -> str:
    if len(refs) == 1:
        return refs[0]
    return ", ".join(refs[:-1]) + " and " + refs[-1]


def make_negative(
    record: Mapping[str, Any], pool: Sequence[PoolSource], rng: random.Random, max_sources: int = 3
) -> Record | None:
    """Same question/memory/history, unrelated sources (or none) -> refusal answer citing nothing."""
    k = rng.randint(0, max_sources)
    if k == 0:
        sources: list[dict[str, str]] = []
        answer = rng.choice(NO_SOURCE_TEMPLATES)
    else:
        picked = _pick_unrelated(record, pool, k, rng)
        if not picked:
            return None
        sources = [dict(p.source) for p in picked]
        refs = list(dict.fromkeys(s["reference"] for s in sources))
        answer = rng.choice(REFUSAL_TEMPLATES).format(refs=_format_refs(refs))

    new = copy.deepcopy(dict(record))
    new.pop("variants", None)
    new["inputs"]["sources"] = sources
    new["answer"] = answer
    new["category"] = "missing_evidence"
    keep = [t for t in record.get("tags", []) if t.startswith("topic:")]
    new["tags"] = sorted(set(keep) | {REFUSAL_TAG, "augmented:negative"})
    new["parent_id"] = record["id"]
    new["id"] = f"{record['id']}__neg{_short_hash(answer, *[s['reference'] for s in sources])}"
    return new


def add_distractors(
    record: Mapping[str, Any], pool: Sequence[PoolSource], rng: random.Random, max_distractors: int = 2
) -> Record | None:
    """Insert 1..max unrelated sources at random positions and renumber answer citations."""
    if not is_rag(record) or is_refusal_record(record):
        return None
    k = rng.randint(1, max_distractors)
    picked = _pick_unrelated(record, pool, k, rng)
    if not picked:
        return None

    original = list(record["inputs"]["sources"])
    tagged: list[tuple[int | None, Mapping[str, str]]] = [(i + 1, s) for i, s in enumerate(original)]
    for p in picked:
        tagged.insert(rng.randint(0, len(tagged)), (None, p.source))
    mapping = {old: new for new, (old, _) in enumerate(tagged, start=1) if old is not None}

    new = copy.deepcopy(dict(record))
    new.pop("variants", None)
    new["inputs"]["sources"] = [dict(s) for _, s in tagged]
    new["answer"] = renumber_citations(record["answer"], mapping)
    new["tags"] = sorted(set(record.get("tags", [])) | {"augmented:distractor"})
    new["parent_id"] = record["id"]
    new["id"] = f"{record['id']}__dis{_short_hash(json.dumps(new['inputs']['sources'], sort_keys=True))}"
    return new


def expand_variants(record: Mapping[str, Any]) -> list[Record]:
    """Base record (without ``variants``) followed by each materialised variant."""
    base = {k: copy.deepcopy(v) for k, v in record.items() if k != "variants"}
    out = [base]
    for i, variant in enumerate(record.get("variants", [])):
        out.append(expand_variant(record, variant, i))
    return out


def record_fingerprint(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(to_messages(record), ensure_ascii=False).encode("utf-8")).hexdigest()


def dedup(records: Iterable[Record]) -> list[Record]:
    seen: set[str] = set()
    out: list[Record] = []
    for record in records:
        fp = record_fingerprint(record)
        if fp not in seen:
            seen.add(fp)
            out.append(record)
    return out
