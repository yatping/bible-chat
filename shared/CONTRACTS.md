# Cross-component contracts

Three components share data and must agree on these formats byte-for-byte:

| Producer | Consumer | Contract |
|---|---|---|
| `content/` (Python) | `ios/` (Swift) | `bible.sqlite` ([schema_content.sql](schema_content.sql)), `rag_embeddings.f16` (§2), FTS query building (§3) |
| `training/` (Python) | `ios/` (Swift) | Prompt format (§1). The model is fine-tuned on exactly the prompt the app builds at runtime |
| `ios/` | `ios/` | `user.sqlite` ([schema_user.sql](schema_user.sql)), memory JSON (§4) |

The golden fixture [fixtures/prompt_golden.json](fixtures/prompt_golden.json) is checked by **both** the Python
tests (`training/tests`) and the Swift tests (`ios/BibleAICore/Tests`). The same applies to
[fixtures/fts_query_golden.json](fixtures/fts_query_golden.json) (`content/tests` and Swift tests),
[fixtures/rrf_golden.json](fixtures/rrf_golden.json), [fixtures/citation_golden.json](fixtures/citation_golden.json) and [fixtures/memory_extraction_golden.json](fixtures/memory_extraction_golden.json).

---

## 1. Prompt format

A request is a list of chat messages, rendered later by Gemma's own chat template
(`processor.apply_chat_template` in training, the GGUF-embedded template via llama.cpp on device).
Never hand-write Gemma control tokens.

```
messages = [
  {role: "system",    content: <system prompt>},          # shared/system_prompt.txt, trailing newline stripped
  ...recent messages, oldest first, raw content, roles "user"/"assistant",
  {role: "user",      content: <context block>}
]
```

`<context block>` is built from up to four sections, **in this order**, joined with a single blank line (`"\n\n"`):

```
MEMORY
[M1] <memory 1 content>
[M2] <memory 2 content>

CONVERSATION SUMMARY
<summary>

SOURCES
[S1] type: <source_type> | reference: <reference>
<text>

[S2] type: <source_type> | reference: <reference>
<text>

QUESTION
<question>
```

Rules:

- All inserted strings are trimmed of leading/trailing whitespace first.
- `MEMORY` section is **omitted entirely** when there are no memories. Memories are numbered `M1..Mn` in the given order.
- `CONVERSATION SUMMARY` section is **omitted entirely** when the summary is nil/empty.
- `SOURCES` section is **always present**. Sources are numbered `S1..Sn` in the given order and separated by one blank line.
  With zero sources the section body is the single line `(no sources retrieved)`.
- `QUESTION` section is always present and last.
- No trailing newline after the question.
- Source ids are assigned by the app, never by the model.

**Chat template.** [fixtures/gemma4_chat_template_golden.json](fixtures/gemma4_chat_template_golden.json) holds the
official Gemma 4 template's rendering (`enable_thinking=False`) of the golden prompts. The app's Swift fallback
template (`ChatTemplate.gemma4`) must reproduce it byte-for-byte; regenerate the fixture if the model revision changes.

**Thought channel.** Base Gemma 4 can open `<|channel>thought … <channel|>` even with thinking disabled. Engines strip
these blocks from generated text before display, storage, citation validation or memory-JSON parsing
(`ThoughtChannelFilter` in Swift, `bible_ft.thinking` in evaluation).

Historical user turns in "recent messages" are stored and replayed as the raw question only (no context block),
historical assistant turns as the final validated answer text.

### Answer format expected from the model

- Plain prose (markdown lists allowed). Citations inline as `[S1]`, multiple as `[S1][S3]`.
- Scripture claims cite scripture sources; commentary-based claims say so ("The commentary notes … [S3]").
- If SOURCES do not support an answer: say the supplied sources do not cover it, and do not cite anything.

### Citation validation (post-generation)

`valid = {S1..Sn}`. First normalise grouped citations: `[S1, S2]` → `[S1][S2]`, ranges `[S2-S4]` → `[S2][S3][S4]`.
Then every `[S<k>]` with k outside 1..n is removed **together with any spaces/tabs immediately before it**.
Result reports `cited` (valid ids, first-appearance order, de-duplicated) and `removed` (invalid ids, first-appearance
order, de-duplicated). See [fixtures/citation_golden.json](fixtures/citation_golden.json).

## 2. Embeddings

- Model: `google/embeddinggemma-300m` (768-dim output).
- Matryoshka truncation: keep the **first 256** dimensions, then L2-normalise. Similarity = dot product.
- Document text prompt: `title: {title} | text: {text}` where title = chunk `reference` (use `none` if empty).
- Query text prompt: `task: search result | query: {text}`.
- Memories are embedded as documents with `title: none | text: {content}`.
- `rag_embeddings.f16`: raw IEEE-754 binary16, little-endian, row-major, `count × dim`, **no header**.
  Row `i` ↔ `embedding_rows.row = i` ↔ `chunk_id`. `count`/`dim` come from `content_meta`.
- Memory vectors in `user.sqlite` are Float32 little-endian blobs (same truncation/normalisation).
- On device, EmbeddingGemma runs as a GGUF through llama.cpp. It must be converted with
  `--sentence-transformers-dense-modules` so the GGUF includes the two Dense projection layers that follow mean
  pooling; otherwise query vectors will not match the document vectors built with sentence-transformers;
  the app applies the prompts above itself because llama.cpp does not add them.

## 3. Retrieval

1. **Explicit reference** — if the query contains a parseable reference (`Matthew 18:22`, `Mt 18`, `1 Cor 13:4-7`),
   look up **scripture** chunks overlapping it (book_index, chapter range, verse range). These form ranked list R,
   ordered by the reference's position in the query, then by chunk position (book, chapter, verse_start). No cap.
2. **Semantic** — query embedding · all document rows, top 30 → list V.
3. **Lexical** — FTS5 BM25, top 30 → list L:
   ```sql
   SELECT rowid, bm25(chunks_fts, 2.0, 3.0, 1.0) AS score
   FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score, rowid LIMIT 30;
   ```
   FTS query construction: lowercase; extract tokens matching `[\p{L}\p{N}]+`; drop tokens in
   [stopwords.txt](stopwords.txt) and tokens of length < 2; de-duplicate keeping first occurrence; wrap each in
   double quotes; join with ` OR `. If no tokens remain, skip lexical search.
4. **Reciprocal rank fusion**, ranks are 1-based, k = 60:
   `score(d) = Σ_lists w_list / (k + rank_list(d))`, weights: R = 2.0, V = 1.0, L = 1.0.
   Ties broken by lower chunk_id. Keep top 6 (configurable), then trim to the sources token budget (2500 tokens,
   `token_estimate`): walk in rank order and stop at the first chunk that would exceed the budget; always keep at
   least the first chunk.

Notes: semantic ties are broken by lower row index. FTS tokens are not Unicode-normalised beyond lowercasing.

## 4. Memory

Extraction output (from `shared/memory_extraction_prompt.txt`), parsed and validated by app code — the model never writes
to the database directly:

```json
{"memories":[{"type":"preference","key":"bible_translation","value":"Preferred Bible translation is WEB.","importance":0.8,"confidence":1.0}]}
```

Validation rules:
- Tolerate surrounding prose / code fences: parse the first balanced `{...}` object.
- `type` ∈ {preference, goal, study_interest, progress, note}, else drop the item.
- `key` ∈ {bible_translation, answer_style, answer_length, explanation_level, theological_tradition, current_study, reading_plan} or null; unknown key → null.
- `value` non-empty after trim, ≤ 300 chars, else drop.
- `importance`, `confidence` clamped to [0,1]; defaults 0.5 / 1.0. Drop items with confidence < 0.5.
- At most 5 items per extraction.

Consolidation:
- Keyed memory, active row exists with same normalised value (case/whitespace-insensitive) → touch `updated_at`.
- Keyed memory, different value → in one transaction set old `superseded=1, superseded_by=new.id`, insert new.
- Unkeyed memory → if an active unkeyed memory has embedding cosine ≥ 0.92, update that row's content instead; else insert.

Retrieval score:
```
score = 0.60 × semantic + 0.25 × importance + 0.15 × recency
recency = 0.5 ^ (age_days / half_life_days)
half_life_days: preference ∞ (recency = 1), note ∞, goal 90, study_interest 30, progress 3
```
Active preferences (keyed, type = preference) are always included; the remainder is filled by score, up to 4 total
memories, minimum score 0.35 for non-preference items.

## 5. Runtime budget (device)

| Item | Tokens |
|---|---:|
| n_ctx | 6144 |
| system | ~300 |
| memory | ≤ 300 |
| conversation summary | ≤ 300 |
| recent messages | ≤ 1000 (messages not covered by the summary, at most 16, trimmed oldest-first to fit) |
| sources | ≤ 2500 |
| answer (max new tokens) | 1200 |

Token estimate everywhere = `ceil(chars / 4)`. Every message is either covered by the rolling summary or replayed.
When more than 16 messages are not yet covered by the summary, fold all but the last 8 into it
(`conversations.summary_message_count` records how many leading messages the summary covers).

Generation defaults: temperature 0.3, top_p 0.9, top_k 40, repeat penalty 1.1. Memory extraction / summarisation:
temperature 0.0, max 256 tokens.
