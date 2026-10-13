#!/usr/bin/env python3
"""Render a Base vs Fine-tuned vs Q4 comparison table from evaluate.py outputs.

Each argument is ``LABEL=PATH`` where PATH is an eval directory or its metrics.json.

Example:
  python training/scripts/compare_reports.py \\
      base=training/outputs/eval/base finetuned=training/outputs/eval/finetuned \\
      q4_k_m=training/outputs/eval/q4_k_m --out training/outputs/eval/comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (section, metric key, display name, higher is better)
ROWS: tuple[tuple[str, str, str, bool], ...] = (
    ("overall", "citation_validity_rate", "Citation validity", True),
    ("overall", "unsupported_citation_rate", "Unsupported citations", False),
    ("overall", "answers_with_invalid_citation_rate", "Answers with invalid ids", False),
    ("overall", "correct_refusal_rate", "Correct refusals", True),
    ("overall", "false_refusal_rate", "False refusals", False),
    ("overall", "citation_coverage", "Citation coverage", True),
    ("overall", "format_adherence", "Format adherence", True),
    ("overall", "n", "Answer items", True),
    ("memory_extraction", "json_valid_rate", "Memory: JSON valid", True),
    ("memory_extraction", "strict_json_rate", "Memory: strict JSON", True),
    ("memory_extraction", "type_key_exact_match", "Memory: type/key exact match", True),
    ("memory_extraction", "empty_accuracy", "Memory: nothing extracted when nothing to extract", True),
    ("summarization", "within_word_limit_rate", "Summary: <= 120 words", True),
    ("summarization", "format_ok_rate", "Summary: format ok", True),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="LABEL=PATH (eval dir or metrics.json)")
    p.add_argument("--out", type=Path, default=None, help="write markdown here as well as stdout")
    return p.parse_args(argv)


def load(spec: str) -> tuple[str, dict[str, Any]]:
    label, _, path_str = spec.partition("=")
    if not path_str:
        raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
    path = Path(path_str)
    if path.is_dir():
        path = path / "metrics.json"
    return label, json.loads(path.read_text(encoding="utf-8"))


def render_table(runs: list[tuple[str, dict[str, Any]]]) -> str:
    header = "| Metric | " + " | ".join(label for label, _ in runs) + " |"
    sep = "|---|" + "---:|" * len(runs)
    lines = [header, sep]
    for section, key, name, higher in ROWS:
        if all(m.get(section, {}).get(key) is None for _, m in runs):
            continue
        cells = []
        for _, m in runs:
            v = m.get(section, {}).get(key)
            cells.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        arrow = "" if key == "n" else (" (higher better)" if higher else " (lower better)")
        lines.append(f"| {name}{arrow} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table = render_table([load(s) for s in args.runs])
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("# Base vs fine-tuned vs quantised\n\n" + table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
