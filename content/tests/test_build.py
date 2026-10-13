import json
import sqlite3

import pytest

from bible_content.build import BuildConfig, build
from bible_content.db import REQUIRED_META_KEYS
from bible_content.embeddings import HashEmbedder, read_f16, sha256_file
from bible_content.fts import build_fts_query
from bible_content.retrieval import LEXICAL_SQL, HybridRetriever


@pytest.fixture(scope="module")
def built(tmp_path_factory, usfm_dir):
    out = tmp_path_factory.mktemp("build")
    commentary = out / "commentary.jsonl"
    long_text = "\n\n".join(" ".join(["Forgiveness is costly and free."] * 30) for _ in range(5))
    commentary.write_text("\n".join(json.dumps(e) for e in [
        {"source_type": "commentary", "title": "On forgiveness", "reference": "Matthew 18:21-35",
         "text": long_text},
        {"source_type": "dictionary", "title": "Talent", "text": "A talent was a large unit of weight."},
    ]) + "\n", encoding="utf-8")
    manifest = build(BuildConfig(usfm_source=usfm_dir, out_dir=out, commentary=[commentary]),
                     HashEmbedder(256))
    return out, manifest


def test_manifest_and_files(built):
    out, manifest = built
    counts = manifest["counts"]
    assert counts["books"] == 2 and counts["scripture_chunks"] == 9 and counts["other_chunks"] >= 3
    emb = out / "rag_embeddings.f16"
    assert emb.stat().st_size == counts["chunks"] * 256 * 2
    assert manifest["files"]["rag_embeddings.f16"]["sha256"] == sha256_file(emb)
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_db_contents(built):
    out, manifest = built
    conn = sqlite3.connect(out / "bible.sqlite")
    meta = dict(conn.execute("SELECT key, value FROM content_meta"))
    assert set(REQUIRED_META_KEYS) <= set(meta)
    assert meta["embedding_sha256"] == sha256_file(out / "rag_embeddings.f16")
    assert meta["embedding_count"] == str(manifest["counts"]["chunks"])
    assert meta["embedding_dim"] == "256" and meta["translation"] == "WEB"
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 66
    assert conn.execute("SELECT book_index FROM book_aliases WHERE alias = '1 cor'").fetchone()[0] == 46
    rows = conn.execute("SELECT row, chunk_id FROM embedding_rows ORDER BY row").fetchall()
    assert [r for r, _ in rows] == list(range(len(rows))) == [c - 1 for _, c in rows]
    assert len(rows) == conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert conn.execute("SELECT text FROM verses WHERE book_index=40 AND chapter=18 AND verse=22"
                        ).fetchone()[0].endswith("until seventy times seven.")
    bad = conn.execute("SELECT count(*) FROM chunks WHERE token_estimate != (length(text) + 3) / 4"
                       ).fetchone()[0]
    assert bad == 0
    kinds = dict(conn.execute("SELECT source_type, count(*) FROM chunks GROUP BY source_type"))
    assert kinds["scripture"] == 9 and kinds["dictionary"] == 1 and kinds["commentary"] >= 2
    for (tokens,) in conn.execute("SELECT token_estimate FROM chunks WHERE source_type='commentary'"):
        assert 200 <= tokens <= 500


def test_fts_finds_pericope(built):
    out, _ = built
    conn = sqlite3.connect(out / "bible.sqlite")
    rows = conn.execute(LEXICAL_SQL, (build_fts_query("seventy times seven"),)).fetchall()
    ref = conn.execute("SELECT reference FROM chunks WHERE id = ?", (rows[0][0],)).fetchone()[0]
    assert ref == "Matthew 18:21-35"
    # Porter stemming: "forgiving" matches "forgive".
    assert conn.execute(LEXICAL_SQL, (build_fts_query("forgiving"),)).fetchall()


def test_embeddings_match_chunks(built):
    out, _ = built
    conn = sqlite3.connect(out / "bible.sqlite")
    vectors = read_f16(out / "rag_embeddings.f16", 256)
    title, text = conn.execute("SELECT reference, text FROM chunks WHERE id = 1").fetchone()
    expected = HashEmbedder(256).embed_documents([(title, text)])[0]
    assert max(abs(a - b) for a, b in zip(vectors[0], expected)) < 1e-3


def test_retriever(built):
    out, _ = built
    r = HybridRetriever(out / "bible.sqlite", None, HashEmbedder(256))
    by_ref = [r.chunks[i].reference for i in r.reference_list("Explain Matthew 18:22")]
    assert by_ref == ["Matthew 18:21-35"]
    assert [r.chunks[i].reference for i in r.reference_list("Mt 18")] == [
        "Matthew 17:24-18:5", "Matthew 18:6-11", "Matthew 18:12-14", "Matthew 18:21-35"]
    top = r.retrieve("Why did Jesus tell Peter to forgive seventy times seven?")
    assert top[0].reference == "Matthew 18:21-35"
    assert r.retrieve("Matthew 18:22")[0].reference == "Matthew 18:21-35"
    assert r.retrieve("The LORD is my shepherd, I shall lack nothing")[0].reference == "Psalm 23:1-6"
    assert r.semantic_list("shepherd green pastures")[0] == next(
        c.id for c in r.chunks.values() if c.reference == "Psalm 23:1-6")
    assert len(r.retrieve("forgive", top_k=6, token_budget=1)) == 1  # always keep at least one
    assert r.reference_list("who is it") == [] and r.lexical_list("who is it") == []  # lexical skipped
