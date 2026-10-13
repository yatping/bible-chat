"""Repository paths."""

from pathlib import Path

CONTENT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = CONTENT_DIR.parent
SHARED_DIR = REPO_DIR / "shared"
SCHEMA_PATH = SHARED_DIR / "schema_content.sql"
RAW_DIR = CONTENT_DIR / "data" / "raw"
BUILD_DIR = CONTENT_DIR / "build"
EVAL_PATH = CONTENT_DIR / "data" / "eval" / "retrieval_eval.jsonl"
