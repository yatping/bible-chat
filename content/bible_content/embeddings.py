"""Embedding backends and the ``rag_embeddings.f16`` format (CONTRACTS.md §2).

Backends:
- ``embeddinggemma``: sentence-transformers ``google/embeddinggemma-300m`` (imported lazily; build
  machine only). The §2 prompt strings are applied here explicitly.
- ``hash``: deterministic feature-hashed bag of words (stdlib only) for tests and offline smoke
  builds. It is NOT semantically meaningful beyond word overlap.

All vectors are truncated to ``dim`` (Matryoshka) and L2-normalised.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from .fts import stopwords

EMBEDDINGGEMMA_MODEL = "google/embeddinggemma-300m"
HASH_MODEL = "hash-bow-v1"
DEFAULT_DIM = 256

Vector = list[float]


def document_prompt(title: str, text: str) -> str:
    return f"title: {title.strip() or 'none'} | text: {text}"


def query_prompt(text: str) -> str:
    return f"task: search result | query: {text}"


def l2_normalise(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else [0.0] * len(vec)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[Vector]:
        """Embed ``(title, text)`` pairs."""
        ...

    def embed_query(self, text: str) -> Vector: ...


class HashEmbedder:
    """Signed feature hashing of lowercase word tokens (stopwords removed, sublinear tf)."""

    _TOKEN_RE = re.compile(r"[^\W_]+")

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self.model_name = HASH_MODEL

    def _embed(self, text: str) -> Vector:
        counts: dict[str, int] = {}
        stop = stopwords()
        for tok in self._TOKEN_RE.findall(text.lower()):
            if len(tok) >= 2 and tok not in stop:
                counts[tok] = counts.get(tok, 0) + 1
        vec = [0.0] * self.dim
        for tok, n in counts.items():
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
            vec[h % self.dim] += (1.0 if (h >> 32) & 1 else -1.0) * (1.0 + math.log(n))
        return l2_normalise(vec)

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[Vector]:
        return [self._embed(f"{title} {text}") for title, text in items]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class EmbeddingGemmaEmbedder:
    """google/embeddinggemma-300m via sentence-transformers (lazy import)."""

    def __init__(self, dim: int = DEFAULT_DIM, model_name: str = EMBEDDINGGEMMA_MODEL,
                 batch_size: int = 32, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on build machine
            raise SystemExit(
                "embeddinggemma backend needs `pip install -r content/requirements.txt` "
                "(sentence-transformers, torch) and access to the gated HF model"
            ) from exc
        self.dim = dim
        self.model_name = model_name
        self.batch_size = batch_size
        # EmbeddingGemma does not support float16 activations; keep float32.
        self._model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> list[Vector]:
        # prompt="" disables any prompt sentence-transformers might add: §2 prompts are explicit.
        out = self._model.encode(texts, batch_size=self.batch_size, prompt="",
                                 convert_to_numpy=True, normalize_embeddings=False,
                                 show_progress_bar=len(texts) > self.batch_size)
        return [l2_normalise(row[: self.dim].tolist()) for row in out]

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[Vector]:
        return self._encode([document_prompt(title, text) for title, text in items])

    def embed_query(self, text: str) -> Vector:
        return self._encode([query_prompt(text)])[0]


def make_embedder(backend: str, dim: int = DEFAULT_DIM, **kwargs: object) -> Embedder:
    if backend == "hash":
        return HashEmbedder(dim)
    if backend == "embeddinggemma":
        return EmbeddingGemmaEmbedder(dim, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown embedder backend: {backend}")


# --- rag_embeddings.f16: raw little-endian IEEE-754 binary16, row-major, no header -------------

def write_f16(path: Path, rows: Iterable[Sequence[float]], dim: int) -> int:
    """Write rows; returns the row count."""
    fmt = struct.Struct(f"<{dim}e")
    count = 0
    with path.open("wb") as f:
        for row in rows:
            if len(row) != dim:
                raise ValueError(f"row {count} has dim {len(row)} != {dim}")
            f.write(fmt.pack(*row))
            count += 1
    return count


def read_f16(path: Path, dim: int) -> list[Vector]:
    data = path.read_bytes()
    row_bytes = 2 * dim
    if len(data) % row_bytes:
        raise ValueError(f"{path}: size {len(data)} is not a multiple of {row_bytes}")
    fmt = struct.Struct(f"<{dim}e")
    return [list(fmt.unpack_from(data, off)) for off in range(0, len(data), row_bytes)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
