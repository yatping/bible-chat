# Offline Bible AI (iOS)

A fully offline Bible study assistant for iPhone. A **QLoRA fine-tuned Gemma 4 E2B** generates answers through
llama.cpp. **EmbeddingGemma** powers local hybrid RAG over the World English Bible. A local **long-term memory**
stores the user's preferences and study context. After the model download, nothing leaves the device.

```text
Fine-tuning  → HOW Gemma answers   (grounded, cites only supplied sources, refuses when evidence is missing)
RAG          → WHAT is relevant    (reference lookup + semantic + BM25, fused with RRF)
Memory       → WHAT it knows about the user (validated JSON memories, consolidation, decay-weighted recall)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Repository layout

| Path | What | Language |
|---|---|---|
| [`shared/`](shared/) | Contracts every component follows: SQLite schemas, prompt format, retrieval and memory rules, prompt texts, golden fixtures | SQL, Markdown, JSON |
| [`content/`](content/README.md) | Builds `bible.sqlite` (verses, chunks, FTS5, cross references) and `rag_embeddings.f16`; retrieval evaluation | Python |
| [`training/`](training/README.md) | Dataset → QLoRA → evaluation → merge → GGUF → Q4_K_M | Python |
| [`ios/BibleAICore/`](ios/README.md) | Swift package: RAG, memory, prompt building, citation validation, orchestration (no dependencies) | Swift |
| [`ios/BibleAI/`](ios/README.md) | SwiftUI app with llama.cpp-backed engines, model download, chat, study, memory settings | Swift |

## Quick start

```bash
# 1. Content package (hash embedder needs no ML deps; use --embedder embeddinggemma for real vectors)
cd content && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_sources.py --cross-references
.venv/bin/python scripts/build_content.py --embedder hash --cross-references data/raw/cross_references.txt --xref-min-votes 5
cd ..

# 2. iOS app (mock model available in DEBUG builds)
ios/scripts/fetch_llama_xcframework.sh
ios/scripts/bundle_content.sh
cd ios && xcodegen generate && open BibleAI.xcodeproj

# 3. Fine-tune on a GPU machine: see training/README.md
```

## Tests

```bash
(cd content  && .venv/bin/python -m pytest -q tests)            # 100 tests
(cd training && .venv/bin/python -m pytest -q tests)            # 131 tests
(cd ios/BibleAICore && swift test)                              # 55 tests
```

The Python and Swift suites are checked against the same golden fixtures in `shared/fixtures/`. The prompt the
model is fine-tuned on is therefore byte-identical to the one the app builds.

## Status

| Verified here | Not yet verified |
|---|---|
| All test suites pass; the app builds for the iOS Simulator and device with the real llama.cpp XCFramework linked | QLoRA training, evaluation and GGUF conversion (needs a CUDA GPU) |
| WEB content build: 31,098 verses, 3,791 chunks, FTS5, 117k cross references | Real EmbeddingGemma vectors (gated model: accept the licence and run `hf auth login`, then rebuild with `--embedder embeddinggemma`) |
| Retrieval eval harness (66 labelled questions) | Inference on a physical iPhone (speed, memory, Gemma 4 chat template end-of-turn handling) |

Known limitations:

- **Chunks aren't true pericopes.** The ebible.org WEB USFM has no section headings, so chunks are ~8-verse
  paragraph windows. Adding a curated heading list would give true pericopes; the chunker already supports headings.
- **Placeholder model manifest.** `ios/BibleAI/Resources/ModelManifest.json` has example URLs and checksums.
  Fill them in from the conversion scripts' output before release.
- **Different llama.cpp tags.** The conversion scripts pin llama.cpp `b11500` and the app runtime pins `b11580`.
  A GGUF from the older converter should load in the newer runtime, but this hasn't been tested.

## Licences of bundled content

- World English Bible: public domain ([ebible.org](https://ebible.org/web/))
- Cross references: [OpenBible.info](https://www.openbible.info/labs/cross-references/), CC-BY
- Models: Gemma and EmbeddingGemma are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
