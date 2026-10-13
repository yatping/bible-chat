from __future__ import annotations

import sys
from pathlib import Path

import pytest

CONTENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONTENT_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
USFM_DIR = FIXTURES / "usfm"


@pytest.fixture(scope="session")
def usfm_dir() -> Path:
    return USFM_DIR


@pytest.fixture(scope="session")
def parsed_books():  # type: ignore[no-untyped-def]
    from bible_content.build import load_usfm

    return load_usfm(USFM_DIR)
