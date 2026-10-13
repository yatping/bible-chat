#!/usr/bin/env python3
"""Validate and preview a commentary/dictionary JSONL, or rebuild the package including it.

Embeddings are computed over all chunks, so adding sources means rebuilding: this script is a thin
wrapper around build_content.py --commentary. JSONL lines:
  {"source_type": "commentary"|"dictionary", "title": "...", "reference": "John 3:16"?, "text": "..."}
Long entries are split into 200-500 token chunks. Only import public-domain or suitably licensed text.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+", help="JSONL files")
    parser.add_argument("--check", action="store_true", help="only validate and print chunk stats")
    args, build_args = parser.parse_known_args()

    from bible_content.sources import load_commentary_jsonl

    for path in args.files:
        chunks = load_commentary_jsonl(path)
        tokens = [c.token_estimate for c in chunks]
        print(f"{path}: {len(chunks)} chunks, tokens min={min(tokens, default=0)} "
              f"max={max(tokens, default=0)}")
    if args.check:
        return
    cmd = [sys.executable, str(Path(__file__).with_name("build_content.py")), *build_args,
           "--commentary", *map(str, args.files)]
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
