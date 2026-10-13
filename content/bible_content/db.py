"""Write ``bible.sqlite`` from shared/schema_content.sql."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .books import ALIASES, BOOKS
from .chunker import Chunk
from .paths import SCHEMA_PATH
from .usfm import ParsedBook

REQUIRED_META_KEYS = (
    "schema_version", "translation", "embedding_model", "embedding_dim", "embedding_count",
    "embedding_file", "embedding_sha256", "built_at",
)


@dataclass(frozen=True)
class CrossReference:
    from_book: int
    from_chapter: int
    from_verse: int
    to_book: int
    to_chapter: int
    to_verse_start: int
    to_verse_end: int
    votes: int


def create_db(path: Path, schema_path: Path = SCHEMA_PATH) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    return conn


def insert_books(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO books (book_index, osis, usfm, name, testament, chapters) VALUES (?,?,?,?,?,?)",
        [(b.index, b.osis, b.usfm, b.name, b.testament, b.chapters) for b in BOOKS],
    )
    conn.executemany("INSERT INTO book_aliases (alias, book_index) VALUES (?,?)", sorted(ALIASES.items()))


def insert_verses(conn: sqlite3.Connection, books: Iterable[ParsedBook], translation: str) -> int:
    rows = [(translation, v.book_index, v.chapter, v.verse, v.text) for b in books for v in b.verses]
    conn.executemany(
        "INSERT INTO verses (translation, book_index, chapter, verse, text) VALUES (?,?,?,?,?)", rows)
    return len(rows)


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> list[int]:
    """Insert chunks with ids 1..n (in order) and map embedding row i -> chunk id i+1."""
    ids = list(range(1, len(chunks) + 1))
    conn.executemany(
        """INSERT INTO chunks (id, source_type, title, reference, translation, book_index, chapter,
           verse_start, chapter_end, verse_end, text, token_estimate)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(i, c.source_type, c.title, c.reference, c.translation, c.book_index, c.chapter,
          c.verse_start, c.chapter_end, c.verse_end, c.text, c.token_estimate)
         for i, c in zip(ids, chunks)],
    )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.executemany("INSERT INTO embedding_rows (row, chunk_id) VALUES (?,?)",
                     [(i - 1, i) for i in ids])
    return ids


def insert_cross_references(conn: sqlite3.Connection, xrefs: Iterable[CrossReference]) -> int:
    rows = [(x.from_book, x.from_chapter, x.from_verse, x.to_book, x.to_chapter,
             x.to_verse_start, x.to_verse_end, x.votes) for x in xrefs]
    conn.executemany(
        """INSERT INTO cross_references (from_book, from_chapter, from_verse, to_book, to_chapter,
           to_verse_start, to_verse_end, votes) VALUES (?,?,?,?,?,?,?,?)""", rows)
    return len(rows)


def write_meta(conn: sqlite3.Connection, meta: dict[str, str]) -> None:
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"missing content_meta keys: {missing}")
    conn.executemany("INSERT OR REPLACE INTO content_meta (key, value) VALUES (?,?)",
                     sorted((k, str(v)) for k, v in meta.items()))


def finalise(conn: sqlite3.Connection) -> None:
    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.execute("VACUUM")
    conn.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
