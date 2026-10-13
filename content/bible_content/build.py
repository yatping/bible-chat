"""End-to-end content build: USFM -> verses + chunks -> bible.sqlite + rag_embeddings.f16 + manifest."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .chunker import Chunk, chunk_books
from .embeddings import Embedder, sha256_file, write_f16
from .sources import load_commentary_jsonl, parse_openbible_xrefs
from .usfm import ParsedBook, parse_usfm

DB_NAME = "bible.sqlite"
EMBEDDINGS_NAME = "rag_embeddings.f16"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "1"


@dataclass
class BuildConfig:
    usfm_source: Path  # directory of .usfm files or the ebible.org zip
    out_dir: Path
    translation: str = "WEB"
    commentary: Sequence[Path] = field(default_factory=list)
    cross_references: Path | None = None
    xref_min_votes: int = 0
    embed_batch: int = 256


def load_usfm(source: Path) -> list[ParsedBook]:
    """Parse the 66 canonical books (front matter, glossary, apocrypha are skipped), in canon order."""
    texts: list[str] = []
    if source.is_dir():
        texts = [p.read_text(encoding="utf-8-sig") for p in sorted(source.iterdir())
                 if p.suffix.lower() in (".usfm", ".sfm")]
    else:
        with zipfile.ZipFile(source) as zf:
            texts = [zf.read(n).decode("utf-8-sig") for n in sorted(zf.namelist())
                     if n.lower().endswith((".usfm", ".sfm"))]
    books: dict[int, ParsedBook] = {}
    for text in texts:
        parsed = parse_usfm(text)
        if parsed is not None and parsed.verses:
            books[parsed.book_index] = parsed
    return [books[i] for i in sorted(books)]


def _embed_all(embedder: Embedder, chunks: Sequence[Chunk], batch: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        vectors.extend(embedder.embed_documents([(c.reference, c.text) for c in part]))
        print(f"  embedded {len(vectors)}/{len(chunks)}", flush=True)
    return vectors


def build(config: BuildConfig, embedder: Embedder) -> dict[str, object]:
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    db_path, emb_path = out / DB_NAME, out / EMBEDDINGS_NAME

    books = load_usfm(config.usfm_source)
    if not books:
        raise SystemExit(f"no canonical USFM books found in {config.usfm_source}")
    chunks: list[Chunk] = chunk_books(books, config.translation)
    n_scripture = len(chunks)
    for path in config.commentary:
        chunks.extend(load_commentary_jsonl(path))

    # Embedding title = chunk reference (CONTRACTS.md §2).
    vectors = _embed_all(embedder, chunks, config.embed_batch)
    count = write_f16(emb_path, vectors, embedder.dim)

    conn = db.create_db(db_path)
    db.insert_books(conn)
    n_verses = db.insert_verses(conn, books, config.translation)
    db.insert_chunks(conn, chunks)
    n_xrefs = 0
    if config.cross_references is not None:
        n_xrefs = db.insert_cross_references(
            conn, parse_openbible_xrefs(config.cross_references, config.xref_min_votes))
    emb_sha = sha256_file(emb_path)
    db.write_meta(conn, {
        "schema_version": SCHEMA_VERSION,
        "translation": config.translation,
        "embedding_model": embedder.model_name,
        "embedding_dim": str(embedder.dim),
        "embedding_count": str(count),
        "embedding_file": EMBEDDINGS_NAME,
        "embedding_sha256": emb_sha,
        "built_at": db.utc_now_iso(),
    })
    db.finalise(conn)

    manifest: dict[str, object] = {
        "translation": config.translation,
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "counts": {
            "books": len(books),
            "verses": n_verses,
            "chunks": len(chunks),
            "scripture_chunks": n_scripture,
            "other_chunks": len(chunks) - n_scripture,
            "cross_references": n_xrefs,
        },
        "files": {
            name: {"bytes": (out / name).stat().st_size, "sha256": sha256_file(out / name)}
            for name in (DB_NAME, EMBEDDINGS_NAME)
        },
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
