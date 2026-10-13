# Architecture

Offline Bible AI runs entirely on an iPhone. Two models live on the device:

| Model | Runtime | Job |
|---|---|---|
| Gemma 4 E2B, QLoRA fine-tuned, Q4_K_M GGUF | llama.cpp (Metal) | Answer questions, extract memories, summarise conversations |
| EmbeddingGemma 300M, Q8_0 GGUF | llama.cpp | Embed queries and memories for semantic search |

The Bible text, chunk embeddings, conversations and memories are all stored locally in SQLite. After the model
download, the app makes no network calls.

## Three jobs, three mechanisms

```text
Fine-tuning  = teaches Gemma HOW to answer      (training/)
RAG          = tells Gemma WHAT is relevant     (content/ builds it, ios/ queries it)
Memory       = tells Gemma WHAT it knows about this user  (ios/, on device)
```

The model is **not** fine-tuned to memorise the Bible. The Bible is in the RAG index. Fine-tuning teaches behaviour:
cite only the sources it was given, refuse when the sources don't cover the question, separate scripture from
commentary, present disputed interpretations neutrally, and follow the user's preferred format.

## Request pipeline

```text
                         SwiftUI ChatView
                               │ question
                               ▼
                        AIOrchestrator (actor)
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
 recent messages +        RAGEngine             MemoryRetriever
 rolling summary              │                      │
 (user.sqlite)       ┌────────┼─────────┐      EmbeddingGemma + scoring
                     ▼        ▼         ▼     (0.6 sem + 0.25 imp + 0.15 recency)
               reference  semantic   FTS5 BM25
               lookup     (vDSP)     (SQLite)
                     └────────┼─────────┘
                              ▼
                     weighted RRF → top 6 → token budget
                              │
                              ▼
                PromptBuilder (shared/CONTRACTS.md §1)
                SYSTEM · MEMORY [M#] · SUMMARY · SOURCES [S#] · QUESTION
                              │
                              ▼
                   Gemma 4 E2B (llama.cpp) ── streaming tokens ──▶ UI
                              │
                              ▼
               CitationValidator (strip ids not in SOURCES)
                              │
                              ▼
                 persist message + cited sources
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
     MemoryExtractor (Gemma, JSON)     ConversationSummarizer
     → Swift validation → consolidation   (when history grows)
     → EmbeddingGemma → user.sqlite
```

RAG and memory retrieval run concurrently (`async let`).

## Build-time pipeline

```text
training/                                      content/
  seed examples (hand-written, WEB quotes)       WEB USFM (ebible.org, public domain)
        │ build_dataset (negatives,                    │ parse → verses
        │   distractors, format variants)              │ chunking (~8-verse passages)  
        ▼                                              ▼
  book-held-out split 80/10/10                   bible.sqlite (verses, chunks, FTS5, aliases)
        │                                              │ EmbeddingGemma "Retrieval-document"
        ▼                                              │ 768 → 256 dims, L2-normalised, f16
  QLoRA on gemma-4-E2B-it (4-bit NF4, r=16,            ▼
  assistant-only loss)                           rag_embeddings.f16 (memory-mapped on device)
        │ evaluate base vs fine-tuned                  │
        ▼                                              ▼
  merge_and_unload → GGUF f16 → Q4_K_M           eval_retrieval: Recall@1/5, MRR
        │ re-evaluate quantised via llama-server
        ▼
  bible-gemma-q4_k_m.gguf  ──download──▶  iPhone  ◀──bundled──  bible.sqlite + rag_embeddings.f16
```

## Why the components agree

The model is trained on the same prompt format the app builds at runtime. `shared/` holds the schemas, prompt
texts and golden fixtures. The Python suites in `training/` and `content/` and the Swift suite in `ios/BibleAICore`
all check against those fixtures, so a format change that isn't made everywhere fails a test.

## Design decisions

- **llama.cpp behind a protocol.** `LLMEngine` and `EmbeddingEngine` hide the runtime. LiteRT-LM, MLX or Core ML
  can replace llama.cpp later without touching RAG, memory or UI code.
- **No vector database.** The WEB build has 3,791 chunks × 256 dims × f16, which is 1.9 MB. A single `cblas_sgemv`
  over a memory-mapped matrix takes milliseconds.
- **Hybrid retrieval.** Semantic search finds paraphrases ("pardoning people" → forgiveness). BM25 and explicit
  reference parsing find exact matches ("Matthew 18:22"). Weighted reciprocal rank fusion combines them.
- **Few, short sources.** A 2B model answers better from 4–8 precise passages than from 30 noisy ones.
- **Citations are app-assigned and app-validated.** The model can't introduce a source the app didn't provide.
- **Memory is structured, not the chat log.** Gemma proposes JSON, Swift validates it, and consolidation supersedes
  stale values (WEB → ASV) while keeping history.
- **6K context, not 128K.** The KV cache and prefill time on a phone dominate. Summaries and RAG keep prompts small.
- **Two databases.** `bible.sqlite` is read-only and replaced on content updates. `user.sqlite` is the user's data.

## Evaluation

| Suite | Where | Measures |
|---|---|---|
| Fine-tuning | `training/scripts/evaluate.py` | valid / unsupported citations, correct refusals, false refusals, format adherence; base vs fine-tuned vs Q4 GGUF |
| Retrieval | `content/scripts/eval_retrieval.py` | Recall@1, Recall@5, MRR for reference / lexical / semantic / hybrid |
| Memory | `ios/BibleAICore/Tests` | preference stored → retrieved after 20 unrelated turns; update supersedes old value |
