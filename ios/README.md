# iOS app: offline Bible AI

A SwiftUI iPhone app (iOS 17+). Everything runs on the device:

- **Answers:** fine-tuned Gemma 4 E2B (Q4_K_M GGUF) via llama.cpp with Metal.
- **Embeddings:** EmbeddingGemma 300M (GGUF) via llama.cpp, used for query and memory vectors.
- **Retrieval:** bundled read-only `bible.sqlite` (FTS5) plus `rag_embeddings.f16`, combined with hybrid RRF.
- **Memory:** writable `user.sqlite` in Application Support holds conversations, rolling summaries and long-term memories.

The only network access is the one-time model download. The binding formats are in [`../shared/CONTRACTS.md`](../shared/CONTRACTS.md).

## Setup

```sh
# 1. llama.cpp XCFramework (pinned tag + sha256 + commit). Release assets only include ios-arm64 + macOS,
#    so the script also builds the iOS Simulator slice from the same tag (needs cmake, ~5 min).
ios/scripts/fetch_llama_xcframework.sh            # --no-simulator for device-only

# 2. Build the content (see content/), then copy it into the app bundle
ios/scripts/bundle_content.sh                     # copies content/build/{bible.sqlite,rag_embeddings.f16}

# 3. Generate and open the Xcode project
cd ios && xcodegen generate && open BibleAI.xcodeproj
```

Other commands:

- Core tests (macOS, no simulator needed): `cd ios/BibleAICore && swift test`
- CLI build: `xcodebuild -project ios/BibleAI.xcodeproj -scheme BibleAI -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`
- After editing anything in `shared/*.txt` or `shared/schema_user.sql`: run `python3 ios/scripts/sync_shared_texts.py`. If you forget, `SharedTextsTests` fails.

You need to set these before shipping:

- `BibleAI/Resources/ModelManifest.json` is a copy of `shared/model_manifest.example.json`. Fill in the real URLs, `sha256` values and `bytes`. Release builds refuse files whose manifest entry has no checksum. DEBUG builds skip the check and log a warning.
- Signing team. The `increased-memory-limit` entitlement also needs the capability enabled on the App ID.

When `bible.sqlite` is missing, the app still builds and runs, and shows a "content not bundled" message. The DEBUG toggle **Use mock model** (Settings, or the setup screen) runs the full UI with scripted answers and no models, which is useful in the simulator.

## Architecture

```
ios/
├── BibleAICore/            Swift package, no dependencies (SQLite3 + Accelerate); all logic, unit-tested on macOS
│   ├── AI/                 ChatMessage, LLMEngine, EmbeddingEngine (+ prompts/truncation/normalisation),
│   │                       GenerationConfig, PromptBuilder (§1), ChatTemplate (Gemma 4 fallback), TokenBudget,
│   │                       UTF8StreamDecoder, AIOrchestrator, MockLLMEngine
│   ├── RAG/                VectorStore (mmap f16 → f32, vDSP_mmul), SemanticSearch, FTSQueryBuilder/FTSRetriever,
│   │                       ReferenceParser, HybridRanker (RRF), CitationValidator, RAGEngine, RetrievedSource
│   ├── Memory/             MemoryExtractor, MemoryStore, MemoryManager (consolidation), MemoryRetriever,
│   │                       ConversationSummarizer
│   ├── Storage/            SQLiteDatabase (prepared-statement cache, transactions), ContentDatabase,
│   │                       UserDatabase, ConversationStore, models
│   ├── Models/             ModelManifest
│   └── Prompts/            SharedTexts.swift: generated verbatim copies of shared/*.txt + schema_user.sql
├── BibleAI/                the app
│   ├── App/                BibleAIApp, AppEnvironment (dependency container)
│   ├── AI/                 GemmaEngine, LlamaEmbeddingEngine (#if canImport(llama))
│   ├── Services/           ModelStore (background download, sha256, no iCloud backup), ContentStore, EngineFactory
│   └── UI/                 Chat, conversation list, Study, Memory settings, model setup, Settings
├── project.yml             xcodegen spec (app → BibleAICore package + Frameworks/llama.xcframework)
└── scripts/                fetch_llama_xcframework.sh, bundle_content.sh, sync_shared_texts.py
```

How one question is answered (`AIOrchestrator.ask`):

1. Wait for the previous turn's background work, so a memory such as "I prefer short answers" applies to the next question.
2. Persist the user message.
3. Embed the query once, then run RAG (reference, semantic and BM25, fused with RRF, trimmed to 2500 tokens) and memory retrieval concurrently with `async let`.
4. Build the §1 prompt: the system prompt, the last 8 messages within 1000 tokens, and the context block with MEMORY, CONVERSATION SUMMARY, SOURCES and QUESTION.
5. Stream tokens to the UI as `.token` events.
6. Validate the citations: grouped citations are normalised, and `[S9]`-style ids that don't match a source are stripped.
7. Persist the answer with its `sources_json`, then emit `.completed`.
8. In the background, extract memories (temperature 0, JSON validated by app code, consolidated in transactions) and update the rolling summary once more than 16 messages fall outside the recent window and are not yet summarised.

### Concurrency

- `GemmaEngine` and `LlamaEmbeddingEngine` are actors with a custom serial executor (`DispatchSerialQueue`). Blocking `llama_decode` calls stay off the main thread and out of the Swift cooperative pool.
- Generation checks `Task.isCancelled` on every token. The Stop button cancels the stream, and the partial answer is still validated and saved.
- The SQLite wrapper is a lock-guarded `Sendable` class.

### Chat template

The contract says to use the template embedded in the GGUF via `llama_chat_apply_template`. llama.cpp's C API only pattern-matches a built-in list of templates, and at b11580 that list covers Gemma 2/3 (`<start_of_turn>`) but not Gemma 4 (`<|turn>role … <turn|>`). So `GemmaEngine` tries the embedded template first, and when that returns < 0 it falls back to `ChatTemplate.gemma4`. The fallback is a pure-Swift port of Google's `chat_template.jinja` for text-only turns with no tools and thinking disabled; its source is cited in the file, and it is unit-tested.

## Runtime budgets and performance

| Item | Value |
|---|---|
| n_ctx | **6144**: the prompt budget (~300 system + 300 memory + 300 summary + 1000 history + 2500 sources) plus 1200 answer tokens. Not 128K: the KV cache would not fit in memory. |
| Answer sampling | temp 0.3, top_p 0.9, top_k 40, repeat penalty 1.1 (last 64 tokens), max 1200 new tokens |
| Extraction and summary | greedy, max 256 tokens |
| GPU | `n_gpu_layers = 999` (all layers on Metal) on device; CPU in the simulator |
| Prompt prefill | `n_batch = n_ubatch = 512` |

Memory use:

| Item | Size |
|---|---|
| Q4_K_M E2B weights | ~3 GB, memory-mapped |
| KV cache at n_ctx 6144 | a few hundred MB |
| EmbeddingGemma Q8_0 | ~300 MB |
| Vector matrix | count × 256 × 4 B (3.8k chunks ≈ 4 MB; 30k ≈ 30 MB) |

- The app enables `com.apple.developer.kernel.increased-memory-limit`. Target devices are iPhones with 8 GB of RAM (15 Pro and later).
- The KV cache is cleared for every request; prompt caching across turns is a possible later optimisation.
- Background memory extraction and summarisation share the single model. The next question waits for them, which takes a few seconds at most.

## Swapping llama.cpp for LiteRT-LM (or MLX / Core ML)

All inference sits behind two protocols in BibleAICore:

```swift
protocol LLMEngine: Sendable {
    func load() async throws
    func generate(messages: [ChatMessage], config: GenerationConfig) -> AsyncThrowingStream<String, Error>
}
protocol EmbeddingEngine: Sendable {
    func rawEmbeddings(for prompts: [String]) async throws -> [[Float]]   // prompts/truncation/normalisation are shared
}
```

To switch:

1. Add a `LiteRTEngine: LLMEngine` that loads the `.litertlm` bundle, applies the model's chat template (or `ChatTemplate.gemma4`), streams decoded text, and stops on cancellation.
2. Return it from `EngineFactory.makeLLM`.
3. Add its manifest entry with the `generator` role.

Nothing else changes: prompts, RAG, memory, persistence and UI only see the protocols. The fine-tuned weights must be exported for the new runtime, and the §1 prompt format stays the same. An embedding swap must still produce EmbeddingGemma vectors, with the Dense layers included (CONTRACTS §2). Otherwise `rag_embeddings.f16` must be rebuilt.

## What is verified where

- `swift test` covers all shared goldens (prompt, citations, FTS query, RRF, memory extraction), the embedded shared texts and schema, VectorStore f16 and top-k, reference parsing, FTS, the full RAGEngine on a small DB of real WEB pericopes, consolidation (WEB → ASV supersede, history, unique-active-key), retrieval scoring and decay, the summariser, the orchestrator end-to-end with a mock LLM (an invalid `[S9]` is stripped), and the "short answers survives 20 turns, then is superseded" scenario. A smoke test runs against `content/build/` when present.
- `xcodebuild` compiles `GemmaEngine` and `LlamaEmbeddingEngine` against the real `llama.h` from the pinned XCFramework, for both the simulator and device.
- Real on-device inference needs the downloaded GGUFs and a physical iPhone. This repo does not test it automatically.
