#!/usr/bin/env python3
"""Build bible.sqlite + rag_embeddings.f16 + manifest.json from WEB USFM (and optional extras)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from bible_content.paths import BUILD_DIR, RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usfm", type=Path, default=RAW_DIR / "engwebp_usfm.zip",
                        help="USFM zip or directory (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=BUILD_DIR, help="output dir (default: %(default)s)")
    parser.add_argument("--embedder", choices=["hash", "embeddinggemma"], default="embeddinggemma")
    parser.add_argument("--dim", type=int, default=256, help="Matryoshka dims kept (default: 256)")
    parser.add_argument("--translation", default="WEB")
    parser.add_argument("--commentary", type=Path, nargs="*", default=[],
                        help="commentary/dictionary JSONL files to include as extra chunks")
    parser.add_argument("--cross-references", type=Path, default=None,
                        help="OpenBible.info cross_references.txt (CC-BY)")
    parser.add_argument("--xref-min-votes", type=int, default=0,
                        help="drop cross references with fewer votes (default: 0)")
    parser.add_argument("--batch-size", type=int, default=32, help="embeddinggemma batch size")
    parser.add_argument("--device", default=None, help="torch device for embeddinggemma (cpu/mps/cuda)")
    args = parser.parse_args()

    from bible_content.build import BuildConfig, build
    from bible_content.embeddings import make_embedder

    kwargs = {"batch_size": args.batch_size, "device": args.device} if args.embedder == "embeddinggemma" else {}
    embedder = make_embedder(args.embedder, args.dim, **kwargs)
    manifest = build(BuildConfig(
        usfm_source=args.usfm, out_dir=args.out, translation=args.translation,
        commentary=args.commentary, cross_references=args.cross_references,
        xref_min_votes=args.xref_min_votes,
    ), embedder)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
