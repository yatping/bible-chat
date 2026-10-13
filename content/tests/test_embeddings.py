import math
import struct

from bible_content.embeddings import (HashEmbedder, document_prompt, query_prompt, read_f16,
                                      write_f16)


def test_prompts():
    assert document_prompt("Matthew 18:21-35", "Then Peter came") == \
        "title: Matthew 18:21-35 | text: Then Peter came"
    assert document_prompt("", "x") == "title: none | text: x"
    assert query_prompt("forgive") == "task: search result | query: forgive"


def test_hash_embedder_deterministic_and_normalised():
    e = HashEmbedder(256)
    a = e.embed_query("seventy times seven forgive")
    assert a == HashEmbedder(256).embed_query("seventy times seven forgive")
    assert len(a) == 256 and math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert e.embed_query("who is it") == [0.0] * 256  # only stopwords -> zero vector


def test_f16_roundtrip(tmp_path):
    e = HashEmbedder(256)
    rows = e.embed_documents([("Genesis 1:1", "In the beginning"), ("John 3:16", "For God so loved")])
    path = tmp_path / "x.f16"
    assert write_f16(path, rows, 256) == 2
    data = path.read_bytes()
    assert len(data) == 2 * 256 * 2  # no header
    assert struct.unpack_from("<e", data, 2)[0] == read_f16(path, 256)[0][1]  # little-endian, row-major
    back = read_f16(path, 256)
    for orig, got in zip(rows, back):
        assert max(abs(a - b) for a, b in zip(orig, got)) < 1e-3
