# content/ — offline RAG content package

Builds, on a Mac/Linux build machine, the read-only content the iPhone app bundles:

| File | What |
|---|---|
| `build/bible.sqlite` | Scripture verses, retrieval chunks, FTS5 index, books + aliases, cross references, build metadata. Schema: [`shared/schema_content.sql`](../shared/schema_content.sql) (executed verbatim). |
| `build/rag_embeddings.f16` | One EmbeddingGemma vector per chunk: raw little-endian IEEE-754 binary16, row-major, `count × dim`, **no header**. Row `i` ↔ `embedding_rows.row = i` ↔ `chunk_id` (currently `chunk_id = row + 1`). |
| `build/manifest.json` | Counts, file sizes and sha256 of the two files above. |

The app runs hybrid retrieval over these files (explicit reference lookup + semantic dot product +
FTS5 BM25, fused with weighted RRF) exactly as specified in [`shared/CONTRACTS.md`](../shared/CONTRACTS.md) §2–§3.
`bible_content/retrieval.py` is a Python mirror of that algorithm used for evaluation.

## Sources and licences

| Source | Licence | Notes |
|---|---|---|
| [World English Bible](https://ebible.org/find/details.php?id=engwebp) (`engwebp_usfm.zip`, ebible.org) | Public domain | 66-book Protestant edition. "World English Bible" is a trademark; the text is PD. |
| [OpenBible.info cross references](https://www.openbible.info/labs/cross-references/) | **CC-BY** | Optional. Attribution required in the app, e.g. "Cross references from OpenBible.info, CC-BY". |
| Commentary / dictionary JSONL (optional) | Must be public domain or compatibly licensed | Nothing is bundled. Candidates: Easton's Bible Dictionary (1897), Matthew Henry's Commentary (1706) — verify the licence of the specific digital edition you import. |

`scripts/download_sources.py` records URL, size and sha256 of each download in `data/raw/sources.json`.

## Setup and commands

```bash
cd content
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # tests + hash builds (stdlib only)
.venv/bin/pip install -r requirements.txt              # real embeddings (torch, sentence-transformers)

.venv/bin/python scripts/download_sources.py --cross-references   # -> data/raw/
.venv/bin/python -m pytest -q tests

# Smoke build (no ML deps, deterministic feature-hashed bag-of-words vectors):
.venv/bin/python scripts/build_content.py --embedder hash \
    --cross-references data/raw/cross_references.txt --xref-min-votes 5

# Production build (needs HF access to google/embeddinggemma-300m: accept the licence, `hf auth login`):
.venv/bin/python scripts/build_content.py --embedder embeddinggemma --device mps \
    --cross-references data/raw/cross_references.txt --xref-min-votes 5

.venv/bin/python scripts/eval_retrieval.py --verbose
```

`build_content.py` options: `--usfm` (zip or directory, default `data/raw/engwebp_usfm.zip`), `--out`
(default `build/`), `--embedder hash|embeddinggemma`, `--dim 256`, `--translation WEB`,
`--commentary FILE...`, `--cross-references FILE`, `--xref-min-votes N`, `--batch-size`, `--device`.
Every script supports `--help` without the ML dependencies.

### Commentary / dictionary sources

JSONL, one entry per line:

```json
{"source_type": "commentary", "title": "On forgiveness", "reference": "Matthew 18:21-35", "text": "..."}
{"source_type": "dictionary", "title": "Talent", "text": "..."}
```

Entries longer than 500 tokens are split (paragraphs, then sentences) into ~200–500 token chunks titled
`Title (n/m)`. `reference` is optional (defaults to the title); when it parses, the chunk's book/chapter/verse
columns are filled too. Because the embedding file covers every chunk, adding sources means a rebuild:

```bash
.venv/bin/python scripts/import_commentary.py my_commentary.jsonl --check        # validate + stats
.venv/bin/python scripts/import_commentary.py my_commentary.jsonl --embedder embeddinggemma  # = build_content.py --commentary ...
```

## How the content is built

- **USFM parsing** (`bible_content/usfm.py`): `\id`, `\c`, `\v` (bridged verses such as `\v 6-7` are stored
  under the first number with `verse_end` on the chunk), section headings `\s`/`\s1..4` (attached to the next
  verse), paragraph / poetry markers (`\p \m \q1 \q2 \b ...`; `\p \m \pi \b \d \ms` etc. mark paragraph or stanza
  starts). Footnotes `\f..\f*`, cross-reference notes `\x..\x*`, word markup `\w word|strong="..."\w*`, and
  character styles (`\wj \add \nd \qs \k \bk ...`) are removed, keeping their text. Titles, intros, Psalm
  superscriptions (`\d`), speaker labels (`\sp`) and `\ms` are not verse text. Only the 66 canonical books are
  kept (front matter and glossary are skipped). Verses that exist only as a footnote in WEB (e.g. Luke 17:36)
  have no text and are omitted.
- **Chunking** (`bible_content/chunker.py`): pericopes of 3–15 verses, never crossing a book. If a book has
  section headings, split at headings (sections may cross a chapter: `Genesis 1:1-2:3`); sections under 3
  verses merge into a neighbour (same chapter preferred); sections over 15 verses split into balanced pieces.
  **The ebible.org WEB USFM has no section headings**, so in practice each chapter is a section, split into
  ~8-verse pieces (DP over cut points, preferring paragraph/stanza starts). Title = heading, else the reference.
- **References**: `Book C:V`, `Book C:V-W`, `Book C1:V1-C2:V2`; the book of Psalms uses the singular
  `Psalm 23:1-6` (all other books use their full name, e.g. `1 Corinthians 13:4-7`).
  `token_estimate = ceil(len(text) / 4)`.
- **Embeddings** (`bible_content/embeddings.py`): document text `title: {reference} | text: {chunk text}`,
  query text `task: search result | query: {q}`; first 256 of 768 dims, then L2-normalised.
  The `hash` backend (`hash-bow-v1`) is for tests/smoke builds only — semantic quality is meaningless.
- **DB** (`bible_content/db.py`): schema file executed verbatim; books, aliases, verses, chunks
  (`id` 1..n), FTS rebuild, `embedding_rows`, cross references, `content_meta` (all required keys including
  `embedding_sha256`), then `PRAGMA optimize` and `VACUUM`.

## Sizes (WEB, current build)

| | |
|---|---:|
| verses | 31,098 (31,103 `\v` markers; 5 footnote-only verses) |
| scripture chunks | 3,791 (avg ≈ 263 tokens, max 625) |
| `rag_embeddings.f16` | 3,791 × 256 × 2 B = 1.94 MB (≈ 20 MB per 40k chunks) |
| `bible.sqlite` without cross references | ≈ 12.6 MB |
| with cross references `--xref-min-votes 5` (117k rows) | ≈ 16.9 MB |
| with all non-negative cross references (344k rows) | ≈ 25.2 MB |

## EmbeddingGemma on the phone (GGUF)

```bash
content/scripts/export_embedder_gguf.sh            # -> content/build/gguf/*.gguf (+ sha256)
VERIFY=1 content/scripts/export_embedder_gguf.sh   # also compares llama.cpp vs sentence-transformers vectors
```

The script pins llama.cpp (`LLAMA_CPP_REF`, default `b11500`), downloads the gated model, converts it to f16
GGUF **with `--sentence-transformers-dense-modules`**, builds `llama-quantize` and quantises to Q8_0.
sentence-transformers' EmbeddingGemma is mean pooling followed by two Dense projections; without those layers
in the GGUF, on-device query vectors would not match the document vectors built here. The app applies the
§2 prompts itself, keeps the first 256 dims and L2-normalises. Re-run `VERIFY=1` whenever either side changes.

## Retrieval evaluation

`data/eval/retrieval_eval.jsonl` holds 66 hand-labelled questions (`kind`: 18 explicit-reference,
22 keyword, 26 paraphrased/semantic), each with relevant ranges in WEB versification. A retrieved chunk counts
as a hit if it overlaps any relevant range. Recall@k = share of questions with a hit in the top k; MRR over the
top 30. Modes: `reference` (list R), `lexical` (L), `semantic` (V), `hybrid` (RRF of R, V, L).

Results with the **hash** embedder (semantic row is not meaningful; rerun after an EmbeddingGemma build):

| mode | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| reference | 0.273 | 0.273 | 0.273 |
| lexical | 0.439 | 0.652 | 0.530 |
| semantic | 0.212 | 0.439 | 0.311 |
| hybrid | 0.697 | 0.864 | 0.766 |

(`reference` can only hit the 18 explicit-reference questions, which it does at rank 1: 18/66 = 0.273.)
