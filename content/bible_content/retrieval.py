"""Python mirror of the app's hybrid retrieval (CONTRACTS.md §3), used for evaluation."""

from __future__ import annotations

import heapq
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .embeddings import Embedder, dot, read_f16
from .fts import build_fts_query
from .references import parse_references
from .rrf import WEIGHT_LEXICAL, WEIGHT_REFERENCE, WEIGHT_SEMANTIC, rrf_fuse

LIST_LIMIT = 30
DEFAULT_TOP_K = 6
DEFAULT_TOKEN_BUDGET = 2500

LEXICAL_SQL = """SELECT rowid, bm25(chunks_fts, 2.0, 3.0, 1.0) AS score
FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score, rowid LIMIT 30;"""

# Scripture chunks overlapping [ (chapter, verse_start) .. (chapter_end, verse_end) ].
REFERENCE_SQL = """SELECT id FROM chunks
WHERE source_type = 'scripture' AND book_index = ?
  AND (chapter < ? OR (chapter = ? AND verse_start <= ?))
  AND (chapter_end > ? OR (chapter_end = ? AND verse_end >= ?))
ORDER BY chapter, verse_start, id;"""


@dataclass(frozen=True)
class ChunkInfo:
    id: int
    source_type: str
    title: str
    reference: str
    book_index: int | None
    chapter: int | None
    verse_start: int | None
    chapter_end: int | None
    verse_end: int | None
    token_estimate: int


class HybridRetriever:
    def __init__(self, db_path: Path, embeddings_path: Path | None, embedder: Embedder | None) -> None:
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        meta = dict(self.conn.execute("SELECT key, value FROM content_meta"))
        self.dim = int(meta["embedding_dim"])
        self.embedder = embedder
        self.chunks: dict[int, ChunkInfo] = {
            row[0]: ChunkInfo(*row) for row in self.conn.execute(
                """SELECT id, source_type, title, reference, book_index, chapter, verse_start,
                   chapter_end, verse_end, token_estimate FROM chunks""")
        }
        self.row_to_chunk = dict(self.conn.execute("SELECT row, chunk_id FROM embedding_rows"))
        path = embeddings_path or db_path.parent / meta["embedding_file"]
        self.vectors = read_f16(path, self.dim) if embedder is not None else []
        if embedder is not None and len(self.vectors) != int(meta["embedding_count"]):
            raise ValueError("embedding_count mismatch between content_meta and embedding file")

    # --- individual lists -------------------------------------------------------------------

    def reference_list(self, query: str) -> list[int]:
        ids: list[int] = []
        for ref in parse_references(query):
            for (chunk_id,) in self.conn.execute(REFERENCE_SQL, (
                    ref.book_index, ref.chapter_end, ref.chapter_end, ref.verse_end,
                    ref.chapter, ref.chapter, ref.verse_start)):
                if chunk_id not in ids:
                    ids.append(chunk_id)
        return ids

    def semantic_list(self, query: str, limit: int = LIST_LIMIT) -> list[int]:
        if self.embedder is None or not self.vectors:
            return []
        q = self.embedder.embed_query(query)
        scored = ((dot(q, vec), self.row_to_chunk[row]) for row, vec in enumerate(self.vectors))
        best = heapq.nsmallest(limit, scored, key=lambda s: (-s[0], s[1]))
        return [chunk_id for _, chunk_id in best]

    def lexical_list(self, query: str) -> list[int]:
        fts_query = build_fts_query(query)
        if fts_query is None:
            return []
        return [row[0] for row in self.conn.execute(LEXICAL_SQL, (fts_query,))]

    # --- fusion -------------------------------------------------------------------------------

    def fused(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[int]:
        return rrf_fuse([
            (WEIGHT_REFERENCE, self.reference_list(query)),
            (WEIGHT_SEMANTIC, self.semantic_list(query)),
            (WEIGHT_LEXICAL, self.lexical_list(query)),
        ], top_k=top_k)

    def apply_token_budget(self, ids: list[int], budget: int = DEFAULT_TOKEN_BUDGET) -> list[int]:
        """Keep the longest prefix within ``budget`` tokens, but always at least one chunk."""
        kept: list[int] = []
        used = 0
        for chunk_id in ids:
            cost = self.chunks[chunk_id].token_estimate
            if kept and used + cost > budget:
                break
            kept.append(chunk_id)
            used += cost
        return kept

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 token_budget: int = DEFAULT_TOKEN_BUDGET) -> list[ChunkInfo]:
        ids = self.apply_token_budget(self.fused(query, top_k), token_budget)
        return [self.chunks[i] for i in ids]

    def close(self) -> None:
        self.conn.close()
