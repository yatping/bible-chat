"""Train / validation / test split with held-out books and no question leakage.

- Test = every record whose ``book`` is in the held-out list (default ``DEFAULT_TEST_BOOKS``),
  so test questions are about passages the model never trained on.
- Remaining records are grouped by normalised question (augmentations of one seed share a
  question) and groups are shuffled into train / validation.
- Any record whose question also appears in test is moved to test, so no question crosses splits.
- Auxiliary-task records (memory_extraction / summarization) have no book: they are split randomly
  by group with the same fractions (``test_fraction`` / ``val_fraction``).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_TEST_BOOKS: tuple[str, ...] = ("Ruth", "Jonah", "James", "Philippians", "Micah")

Record = dict[str, Any]


def normalise_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def question_key(record: Mapping[str, Any]) -> str:
    """Grouping key: the question for RAG records, the input text for auxiliary tasks."""
    inputs = record["inputs"]
    task = record.get("task", "rag_answer")
    if task == "memory_extraction":
        return "mem:" + normalise_question(inputs["user_message"])
    if task == "summarization":
        return "sum:" + normalise_question(" ".join(m["content"] for m in inputs["messages"]))
    return normalise_question(inputs["question"])


def _split_aux(records: Sequence[Record], val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[Record]]:
    """Random group split per task, so each task with >= 3 groups gets >= 1 test and >= 1 val group."""
    out: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    for task in sorted({r["task"] for r in records}):
        groups: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            if r["task"] == task:
                groups[question_key(r)].append(r)
        keys = sorted(groups)
        random.Random(f"{seed}:{task}").shuffle(keys)
        minimum = 1 if len(keys) >= 3 else 0
        n_test = max(minimum, round(test_fraction * len(keys)))
        n_val = max(minimum, round(val_fraction * len(keys)))
        for i, key in enumerate(keys):
            name = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[name].extend(groups[key])
    return out


def split_records(
    records: Sequence[Record],
    test_books: Iterable[str] = DEFAULT_TEST_BOOKS,
    val_fraction: float = 0.1,
    seed: int = 13,
    aux_test_fraction: float = 0.1,
) -> dict[str, list[Record]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}``.

    ``val_fraction`` is relative to the full dataset (80/10/10 by default when the held-out books
    make up roughly 10%).
    """
    aux = [r for r in records if r.get("task", "rag_answer") != "rag_answer"]
    aux_splits = _split_aux(aux, val_fraction, aux_test_fraction, seed)
    records = [r for r in records if r.get("task", "rag_answer") == "rag_answer"]

    held_out = {b.strip() for b in test_books if b.strip()}
    test = [r for r in records if r["book"] in held_out]
    test_questions = {question_key(r) for r in test}

    rest: list[Record] = []
    for r in records:
        if r["book"] in held_out:
            continue
        if question_key(r) in test_questions:
            test.append(r)
        else:
            rest.append(r)

    groups: dict[str, list[Record]] = defaultdict(list)
    for r in rest:
        groups[question_key(r)].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target_val = round( val_fraction * len(records))
    val: list[Record] = []
    train: list[Record] = []
    for key in keys:
        (val if len(val) < target_val else train).extend(groups[key])
    return {name: part + aux_splits[name] for name, part in (("train", train), ("val", val), ("test", test))}


def leaked_questions(splits: Mapping[str, Sequence[Record]]) -> set[str]:
    """Normalised questions present in more than one split (should be empty)."""
    owner: dict[str, str] = {}
    leaked: set[str] = set()
    for name, recs in splits.items():
        for r in recs:
            q = question_key(r)
            if owner.setdefault(q, name) != name:
                leaked.add(q)
    return leaked
