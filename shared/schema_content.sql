-- Content database (bible.sqlite)
-- Built offline by content/ pipeline, shipped read-only inside the iOS app bundle.
-- Single source of truth: both the Python builder and the Swift reader rely on this file.

PRAGMA user_version = 1;

-- Build metadata. Required keys:
--   schema_version   "1"
--   translation      e.g. "WEB"
--   embedding_model  e.g. "google/embeddinggemma-300m"
--   embedding_dim    e.g. "256"
--   embedding_count  number of rows in rag_embeddings.f16
--   embedding_file   "rag_embeddings.f16"
--   embedding_sha256 hex sha256 of the embedding file
--   built_at         ISO-8601 UTC timestamp
CREATE TABLE content_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 66-book Protestant canon, book_index 1 (Genesis) .. 66 (Revelation).
CREATE TABLE books (
    book_index INTEGER PRIMARY KEY,
    osis       TEXT NOT NULL UNIQUE,   -- "Matt"
    usfm       TEXT NOT NULL UNIQUE,   -- "MAT"
    name       TEXT NOT NULL,          -- "Matthew"
    testament  TEXT NOT NULL CHECK (testament IN ('OT', 'NT')),
    chapters   INTEGER NOT NULL
);

-- Normalised aliases used by reference parsing (lowercase, single spaces, no dots):
-- "matthew", "matt", "mt", "1 corinthians", "1 cor", "1cor", "first corinthians", "song of songs", ...
CREATE TABLE book_aliases (
    alias      TEXT PRIMARY KEY,
    book_index INTEGER NOT NULL REFERENCES books(book_index)
);

-- Exact verse text, for precise lookups and quoting.
CREATE TABLE verses (
    translation TEXT    NOT NULL,
    book_index  INTEGER NOT NULL REFERENCES books(book_index),
    chapter     INTEGER NOT NULL,
    verse       INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    PRIMARY KEY (translation, book_index, chapter, verse)
) WITHOUT ROWID;

-- Retrieval units. Scripture chunks are pericopes (3–15 verses, usually one section heading);
-- commentary / dictionary chunks are 200–500 tokens.
CREATE TABLE chunks (
    id             INTEGER PRIMARY KEY,
    source_type    TEXT NOT NULL CHECK (source_type IN ('scripture', 'commentary', 'dictionary', 'cross_reference')),
    title          TEXT NOT NULL,          -- section heading, or the reference if none
    reference      TEXT NOT NULL,          -- canonical display reference: "Matthew 18:21-35", "Genesis 1:1-2:3"
    translation    TEXT,                   -- "WEB" for scripture, NULL otherwise
    book_index     INTEGER REFERENCES books(book_index),
    chapter        INTEGER,                -- start chapter
    verse_start    INTEGER,
    chapter_end    INTEGER,                -- end chapter (== chapter unless the chunk crosses a chapter boundary)
    verse_end      INTEGER,
    text           TEXT NOT NULL,          -- plain text, no markup, verses joined by single spaces
    token_estimate INTEGER NOT NULL        -- ceil(len(text) / 4)
);
CREATE INDEX idx_chunks_ref ON chunks (book_index, chapter, verse_start);

-- Lexical retrieval (BM25). External-content table over chunks.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    title,
    reference,
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'porter unicode61 remove_diacritics 2'
);

-- Row i of rag_embeddings.f16  <->  chunk_id.
CREATE TABLE embedding_rows (
    row      INTEGER PRIMARY KEY,          -- 0-based, contiguous
    chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(id)
);

-- Optional: verse-level cross references (e.g. OpenBible.info, CC-BY).
CREATE TABLE cross_references (
    from_book      INTEGER NOT NULL,
    from_chapter   INTEGER NOT NULL,
    from_verse     INTEGER NOT NULL,
    to_book        INTEGER NOT NULL,
    to_chapter     INTEGER NOT NULL,
    to_verse_start INTEGER NOT NULL,
    to_verse_end   INTEGER NOT NULL,
    votes          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_xref_from ON cross_references (from_book, from_chapter, from_verse);
