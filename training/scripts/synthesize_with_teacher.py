#!/usr/bin/env python3
"""Draft extra Q/A records from content DB chunks with a teacher model (optional).

For each sampled scripture chunk:
  1. ask the teacher for one natural user question the passage answers;
  2. build the exact runtime prompt (system prompt + SOURCES incl. optional distractor chunks)
     and ask the teacher to answer it, citing [S#].

Every draft is written with ``"needs_review": true``. A human must check it (accuracy, neutrality,
citations) and flip it to ``false``; build_dataset.py ignores unreviewed drafts unless
``--include-unreviewed`` is passed. Uses only the standard library (urllib), so any
OpenAI-compatible endpoint works (OpenAI, vLLM, llama-server, Ollama's /v1, ...).

Example:
  export TEACHER_API_KEY=...
  python training/scripts/synthesize_with_teacher.py --base-url https://api.openai.com/v1 \\
      --model gpt-4.1 --limit 50 --out training/outputs/teacher_drafts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from bible_ft import REPO_ROOT, TRAINING_DIR
from bible_ft.aux_tasks import first_json_object as find_json_object
from bible_ft.citations import validate_citations
from bible_ft.jsonl import write_jsonl
from bible_ft.prompt_format import build_messages
from bible_ft.schema import validate_record

QUESTION_PROMPT = """You write questions for a Bible study app.
Given the passage below, write ONE natural question a user might ask that this passage answers directly.
Do not mention the passage reference unless a user naturally would. Return JSON only:
{{"question": "...", "category": "direct_scripture" | "explanation"}}

REFERENCE: {reference}
PASSAGE: {text}"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="TEACHER_API_KEY", help="env var holding the API key")
    p.add_argument("--content-db", type=Path, default=REPO_ROOT / "content" / "build" / "bible.sqlite")
    p.add_argument("--limit", type=int, default=50, help="number of chunks to draft from")
    p.add_argument("--max-distractors", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=Path, default=TRAINING_DIR / "outputs" / "teacher_drafts.jsonl")
    return p.parse_args(argv)


def chat(base_url: str, model: str, api_key: str | None, messages: list[dict[str, str]], temperature: float) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "temperature": temperature}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


def first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first balanced ``{...}`` (tolerates prose / code fences around it)."""
    candidate = find_json_object(text)
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_chunks(db: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT c.id, c.reference, c.text, b.name FROM chunks c JOIN books b ON b.book_index = c.book_index "
            "WHERE c.source_type = 'scripture' AND c.token_estimate <= 400 ORDER BY c.id"
        ).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "reference": r[1], "text": r[2], "book": r[3]} for r in rows]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.content_db.exists():
        raise SystemExit(f"content DB not found: {args.content_db}")
    api_key = os.environ.get(args.api_key_env)
    rng = random.Random(args.seed)
    chunks = load_chunks(args.content_db)
    sample = rng.sample(chunks, min(args.limit, len(chunks)))

    drafts: list[dict[str, Any]] = []
    skipped = 0
    for chunk in sample:
        try:
            q_raw = chat(args.base_url, args.model, api_key, [{"role": "user", "content": QUESTION_PROMPT.format(**chunk)}], args.temperature)
            q = first_json_object(q_raw) or {}
            question = str(q.get("question", "")).strip()
            category = q.get("category") if q.get("category") in {"direct_scripture", "explanation"} else "direct_scripture"
            if not question:
                skipped += 1
                continue

            others = [c for c in chunks if c["book"] != chunk["book"]]
            sources = [{"type": "scripture", "reference": c["reference"], "text": c["text"]}
                       for c in rng.sample(others, rng.randint(0, args.max_distractors))]
            sources.insert(rng.randint(0, len(sources)), {"type": "scripture", "reference": chunk["reference"], "text": chunk["text"]})
            inputs = {"memories": [], "summary": None, "sources": sources, "recent_messages": [], "question": question}
            answer_raw = chat(args.base_url, args.model, api_key,
                              build_messages([], None, sources, [], question), args.temperature)
        except (OSError, KeyError, ValueError) as exc:
            print(f"chunk {chunk['id']}: request failed: {exc}", file=sys.stderr)
            skipped += 1
            continue

        answer = validate_citations(answer_raw, len(sources)).text.strip()
        record = {
            "id": f"teacher-{chunk['id']}",
            "category": category,
            "book": chunk["book"],
            "inputs": inputs,
            "answer": answer,
            "tags": ["teacher_draft"],
            "needs_review": True,
        }
        if errors := validate_record(record):
            print(f"chunk {chunk['id']}: draft rejected: {errors}", file=sys.stderr)
            skipped += 1
            continue
        drafts.append(record)
        print(f"drafted {record['id']}: {question}")

    write_jsonl(args.out, drafts)
    print(f"wrote {len(drafts)} drafts (skipped {skipped}) -> {args.out}; review them and set needs_review=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
