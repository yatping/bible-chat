-- User database (user.sqlite)
-- Created on device in Application Support/. Writable. Never leaves the phone.

PRAGMA user_version = 1;
PRAGMA foreign_keys = ON;

-- Long-term memory. One row per durable fact.
--   type: 'preference' | 'goal' | 'study_interest' | 'progress' | 'note'
--   memory_key: set for "slot" memories that must be consolidated (one active value per key),
--               NULL for free-form semantic memories.
CREATE TABLE memories (
    id                TEXT PRIMARY KEY,          -- UUID
    type              TEXT NOT NULL CHECK (type IN ('preference', 'goal', 'study_interest', 'progress', 'note')),
    memory_key        TEXT,
    content           TEXT NOT NULL,             -- natural-language statement injected into the prompt
    importance        REAL NOT NULL DEFAULT 0.5, -- 0..1
    confidence        REAL NOT NULL DEFAULT 1.0, -- 0..1
    source_message_id TEXT,                      -- message the memory was extracted from
    created_at        REAL NOT NULL,             -- unix seconds
    updated_at        REAL NOT NULL,
    last_accessed_at  REAL,
    superseded        INTEGER NOT NULL DEFAULT 0,
    superseded_by     TEXT REFERENCES memories(id)
);

-- Consolidation invariant: at most one active memory per key.
-- Updating a keyed memory = mark old row superseded + insert new row, in one transaction
-- (history is preserved for the Memory settings screen).
CREATE UNIQUE INDEX idx_memories_active_key
    ON memories (memory_key)
    WHERE memory_key IS NOT NULL AND superseded = 0;

-- EmbeddingGemma vector per memory: little-endian Float32, `dim` values, L2-normalised.
CREATE TABLE memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL
);

CREATE TABLE conversations (
    id                 TEXT PRIMARY KEY,
    title              TEXT,
    summary            TEXT,                     -- rolling summary of messages older than the recent window
    summary_message_count INTEGER NOT NULL DEFAULT 0, -- how many messages the summary covers
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,               -- raw text (user question / validated assistant answer)
    sources_json    TEXT,                        -- assistant only: JSON array of cited sources [{id,reference,type,chunk_id}]
    created_at      REAL NOT NULL
);
CREATE INDEX idx_messages_conversation ON messages (conversation_id, created_at);
