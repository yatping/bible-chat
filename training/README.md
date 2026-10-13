# training/ — fine-tune Gemma 4 E2B for grounded Bible answers

The fine-tune teaches **behaviour**, not Bible content. The Bible lives in the RAG index
(`content/` → `bible.sqlite`); at runtime the app retrieves passages and passes them in as
numbered `SOURCES`. Fine-tuning teaches the model what to do with them:

- answer only from the supplied sources, with inline `[S1]` citations to ids that exist;
- say so, and cite nothing, when the sources don't cover the question;
- keep scripture, commentary and interpretation apart ("The commentary notes … [S3]");
- present disputed doctrines (predestination, baptism, the Lord's Supper, the millennium) neutrally;
- follow the requested format and length, including preferences from `MEMORY`. Memory is context, not
  instructions, so "never cite" in memory doesn't switch citations off.

Memorising verses would be pointless: a 2B model would misquote them, and every fact it needs is
already in the prompt. So the examples are short, the sources vary, and the training signal is on
how the model answers.

**Training data == runtime prompt.** Every example is rendered with `bible_ft.prompt_format`, which
reproduces `shared/fixtures/prompt_golden.json` byte-for-byte (shared/CONTRACTS.md §1), and then with
Gemma's own `processor.apply_chat_template`. On device, llama.cpp renders the same messages with the
template embedded in the GGUF.

## Layout

```
bible_ft/            pure Python, no torch: prompt_format, aux_tasks, citations, schema, metrics, augment, split
data/seed/*.jsonl    55 hand-written answer seeds (+5 memory/format variants, 10 categories),
                     25 memory-extraction seeds, 8 summarisation seeds
scripts/             build_dataset → split_dataset → train_qlora → evaluate → merge_adapter → convert_gguf.sh
tests/               pytest (golden fixtures, schema, augmentation, split, metrics, collator, --help)
outputs/             generated, gitignored
```

## 0. Setup

```bash
# Laptop (data + tests only, no GPU needed)
python3 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements-dev.txt
training/.venv/bin/python -m pytest training/tests -q

# GPU box (RunPod / Lambda / Colab): Python 3.11–3.13, CUDA 12.x
pip install -r training/requirements.txt
huggingface-cli login        # accept the Gemma licence for google/gemma-4-E2B-it first
```

**Hardware.** One GPU with **24 GB or more** (L4, A10G, RTX 4090/3090, A100). The 4-bit base model is
small. Most of the memory goes to `modules_to_save=["lm_head","embed_tokens"]`: with Gemma's 262k
vocabulary, that is a full trainable copy of the embedding plus its optimizer state, about 6–8 GB.
On a 16 GB T4 (Colab free tier), pass `--no-modules-to-save --dtype fp16`. With the seed-only dataset
(about 170 training rows), 3 epochs take minutes. With a few thousand rows, expect under an hour on an
A100. Merging needs about 12 GB of CPU RAM for the bf16 base model.

## 1. Build the dataset

```bash
python training/scripts/build_dataset.py --seed 13
# -> training/outputs/dataset.jsonl + dataset_stats.json
```

The script loads the seeds and validates every one (an invalid seed is a hard error), then expands them:

| Step | What it does |
|---|---|
| variants | Seed `variants` add memory such as "The user prefers short answers" with a matching answer. These exist only where a seed author wrote one. |
| negatives | Each answerable question is paired with *unrelated* sources, or none, and gets a refusal answer that cites nothing. The wording varies, and the answer names what the sources do cover. "Unrelated" means a different `topic:` tag, a different book, and no shared content words. |
| distractors | 1–2 irrelevant sources are inserted at random positions, and the answer's `[S#]` ids are renumbered to match. |
| teacher drafts | `--teacher-drafts FILE` merges drafts with `needs_review: false`. Pass `--include-unreviewed` to also take unreviewed ones. |
| dedup + validate | Identical rendered conversations are dropped, then every record is re-validated with `bible_ft.schema`. |

If `content/build/bible.sqlite` exists, the script also samples short scripture chunks from it as
distractors (use `--no-content-db` to turn this off). Held-out test books are never used as
distractors. Output is deterministic for a given `--seed`. The default of 1 negative per answerable
record keeps refusals at about 28% of the data. Raising it risks teaching the model to over-refuse.

**Seed categories:** direct_scripture, explanation, comparison (Matthew vs Luke), historical_context
(Pharisees, Samaritans), theme_search, source_grounding (whale vs "huge fish", how many wise men),
missing_evidence (refusals), conflicting_interpretations, formatting (short / bullets / study notes /
detailed / one sentence), distractor_citation.

Scripture texts are World English Bible (public domain), copied from the same edition as the content
DB so that training sources match what the app retrieves (`test_seed_scripture_matches_content_db`
checks this when the DB is present). Commentary and dictionary sources are short, neutral notes written
for this dataset and labelled as such. Every quoted span in an answer is checked against its sources
(`test_answer_quotes_come_from_sources`).

### Auxiliary tasks: memory extraction and summarisation

The same on-device model also extracts memories and maintains the conversation summary. Training
only on answers could degrade those skills, so the dataset includes seeds for both tasks:

- `data/seed/memory_extraction.jsonl`: 25 seeds. 15 extract something: translation, answer style
  and length, explanation level, tradition, current study, goals, reading progress, explicit notes,
  and one multi-item message. 10 extract nothing: casual questions, greetings, Bible-knowledge
  statements, a fact the assistant said earlier, one-off "just this time" requests, and a remark
  about a friend's church, where the right answer is not to infer anything.
- `data/seed/summarization.jsonl`: 8 seeds, each a single paragraph of 120 words or fewer. They
  cover dropping greetings, folding in a previous summary, open questions, and a message over 600
  characters that gets truncated.

These records have `"task": "memory_extraction" | "summarization"`. Records without the field are
`rag_answer`. Auxiliary records also have `category == task`, and `book` is optional.
`build_dataset.py` passes them through unchanged (no negatives or distractors). `split_dataset.py`
splits them randomly per task: 10% test and 10% validation, with at least one record of each (`--aux-test-fraction`).

**Exact message shape.** This must match `ios/BibleAICore` `MemoryExtractor` and
`ConversationSummarizer`: a **single user turn** containing the filled template, with **no system
message**. The model's reply is the assistant turn.

| Task | User content |
|---|---|
| memory_extraction | `shared/memory_extraction_prompt.txt` with trailing newlines stripped, `{{user_message}}` → user message trimmed |
| summarization | `shared/summarization_prompt.txt` with trailing newlines stripped, `{{previous_summary}}` → trimmed previous summary or `(none)` if empty, `{{messages}}` → one line per message, `User: <content>` / `Assistant: <content>` (content trimmed, cut to 600 characters + `…`), joined with `\n` |

```json
{"id": "seed-mem-translation-web", "task": "memory_extraction", "category": "memory_extraction",
 "inputs": {"user_message": "Please always quote the WEB translation when you answer."},
 "answer": "{\"memories\":[{\"type\":\"preference\",\"key\":\"bible_translation\",\"value\":\"Preferred Bible translation is WEB.\",\"importance\":0.8,\"confidence\":1.0}]}",
 "tags": ["extract"]}
{"id": "seed-sum-…", "task": "summarization", "category": "summarization",
 "inputs": {"previous_summary": null, "messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]},
 "answer": "The user …", "tags": []}
```

Validation rules (`bible_ft.aux_tasks`):

- A memory answer must be **compact, strict JSON**, exactly `{"memories":[...]}`, with at most 5
  items. Each item has exactly `type`, `key`, `value`, `importance` and `confidence`, and must
  survive the CONTRACTS §4 rules unchanged: allowed type, allowed key or null, trimmed non-empty
  value of 300 characters or fewer, numbers in [0,1], confidence ≥ 0.5.
- The app-side parser (`parse_memories`) reproduces `shared/fixtures/memory_extraction_golden.json`.
- A summary must be 120 words or fewer, one paragraph, with no `[S#]`/`[M#]` ids and no preamble.

### Record format

```json
{"id": "seed-ground-jonah-whale", "category": "source_grounding", "book": "Jonah",
 "inputs": {"memories": [], "summary": null,
            "sources": [{"type": "scripture", "reference": "Jonah 1:17", "text": "The LORD prepared a huge fish …"}],
            "recent_messages": [], "question": "Was Jonah swallowed by a whale?"},
 "answer": "The supplied passage speaks of “a huge fish” … [S1] …",
 "tags": ["topic:jonah"]}
```

`schema.validate_record` enforces these rules:

- Answer citations must be a subset of the supplied `S1..Sn`, in canonical `[S1][S2]` form, with no `[M#]` and no control tokens.
- Refusals (tag `refusal`) cite nothing and must be detected by the metrics refusal heuristic.
- Answerable records cite at least one source.
- History alternates roles and ends with an assistant turn.

`format:*` tags drive the format-adherence metric.

### Optional: teacher drafts

```bash
export TEACHER_API_KEY=...
python training/scripts/synthesize_with_teacher.py --base-url https://api.openai.com/v1 --model <model> --limit 100
# Review training/outputs/teacher_drafts.jsonl by hand. Fix or delete each draft, then set "needs_review": false.
python training/scripts/build_dataset.py --teacher-drafts training/outputs/teacher_drafts.jsonl
```

Any OpenAI-compatible endpoint works, including a local `llama-server`. The script uses only the
standard library.

## 2. Split

```bash
python training/scripts/split_dataset.py --test-books Ruth,Jonah,James,Philippians,Micah
# -> outputs/splits/{train,val,test}.jsonl + split_report.json   (seed build: 196 / 25 / 28, incl. auxiliary tasks)
```

The test split is every record from the **held-out books**, so evaluation measures behaviour on
passages the model never saw in training. Validation is a random 10% of the rest, grouped by
normalised question so that augmentations of one seed never cross splits. The script fails on any
question leakage.

## 3. QLoRA training (GPU)

```bash
python training/scripts/train_qlora.py --dry-run          # renders + shows exactly which tokens are learned
python training/scripts/train_qlora.py \
  --train training/outputs/splits/train.jsonl --val training/outputs/splits/val.jsonl \
  --output-dir training/outputs/gemma4-e2b-bible-qlora
tensorboard --logdir training/outputs/gemma4-e2b-bible-qlora
```

The script follows Google's Gemma QLoRA recipe:

- `AutoModelForMultimodalLM` + `AutoProcessor`.
- NF4 4-bit quantization with double quant, bf16 compute (fp16 on GPUs without bf16 support).
- `prepare_model_for_kbit_training`.
- LoRA with `r=16, alpha=16, dropout=0.05`, `modules_to_save=["lm_head","embed_tokens"]` and `ensure_weight_tying=True`.
- SFT settings: `max_length=2048`, 3 epochs, batch size 1 × grad accum 8, lr 2e-4 (constant), eval and save every epoch, TensorBoard.
- `skip_prepare_dataset` and `remove_unused_columns=False`, because a custom collator does the tokenisation.

The collator masks every token with `-100` except the **final assistant answer** and its end-of-turn
marker. That covers the system prompt, the replayed `recent_messages` (earlier answers included) and
the context block.

By default, LoRA targets only the language model's attention and MLP projections. Pass
`--target-modules all-linear` to also adapt the vision and audio towers, as in the generic recipe.
Every hyperparameter has a CLI flag (`--help`).

Prompts are kept short, so the 2048-token training length covers the seed data. The device budget is
6144 tokens. If you add long multi-turn or many-source examples, raise `--max-length`. Examples whose
prompt exceeds it trigger a warning and teach nothing.

## 4. Evaluate base vs fine-tuned

```bash
python training/scripts/evaluate.py --backend hf --model google/gemma-4-E2B-it --label base
python training/scripts/evaluate.py --backend hf --model google/gemma-4-E2B-it \
  --adapter training/outputs/gemma4-e2b-bible-qlora/adapter --label finetuned
```

Evaluation uses the runtime prompt and the §5 generation defaults: temperature 0.3, top_p 0.9,
top_k 40, repeat penalty 1.1, max 1200 tokens. Add `--greedy` for low-noise A/B runs. Outputs are
citation-validated exactly as on device. Each run writes `predictions.jsonl`, `metrics.json` and a
`report.md` with per-category tables and failing examples.

Metrics (`bible_ft/metrics.py`):

| Metric | Definition |
|---|---|
| citation validity | in-range `[S#]` / all `[S#]` emitted |
| unsupported citations | valid citations to a source the gold answer doesn't cite (distractors, or any citation on a refusal item) / valid citations |
| correct refusal rate | refusals on items whose gold answer is a refusal |
| false refusal rate | refusals on answerable items |
| citation coverage | answerable items whose prediction cites ≥ 1 valid source |
| format adherence | prediction passes its `format:*` check (short ≤ 90 words, bullets ≥ 2, study notes, detailed ≥ 120 words, one sentence) |
| memory: JSON valid / strict JSON | the app can parse a `{"memories":[…]}` object / the whole output is exactly that JSON |
| memory: type/key exact match | the multiset of (type, key) of app-validated items equals the gold's. Also reported: `empty_accuracy` (nothing extracted when the gold is empty) and `extraction_recall` |
| summary: ≤ 120 words / format ok | word limit, and the summary format rules above |

Memory-extraction and summarisation items use the §5 settings: temperature 0.0 and a 256-token cap.

**Refusal heuristic:** an answer counts as a refusal when it contains a refusal phrase (for example
"the supplied sources do not cover", "not enough information", "I can't answer … from the sources")
**and** has no valid citation. A caveat inside a cited answer is not a refusal.

## 5. Merge → GGUF → Q4_K_M

```bash
python training/scripts/merge_adapter.py --adapter training/outputs/gemma4-e2b-bible-qlora/adapter \
  --out training/outputs/merged                     # bf16 merge_and_unload, 2 GB safetensors shards + processor
SMOKE=1 training/scripts/convert_gguf.sh   # LLAMA_CPP_REF defaults to b11500 training/outputs/merged training/outputs/gguf
```

`convert_gguf.sh` does the following:

1. Clones llama.cpp and checks out `LLAMA_CPP_REF` (default `b11500`, the same tag `content/scripts/export_embedder_gguf.sh` uses). Keep it in sync with the llama.cpp version the iOS app links.
2. Builds `llama-quantize`.
3. Runs `convert_hf_to_gguf.py` to produce an f16 GGUF, which embeds the chat template.
4. Runs `llama-quantize` to produce Q4_K_M (about 3 GB for E2B; the script prints the exact size).
5. Prints the byte size and sha256 of both files and writes `*.manifest.json`, which supplies the values for `shared/model_manifest.example.json`.

With `SMOKE=1`, it also runs a short `llama-cli` prompt.

Check that the template survived the conversion:

```bash
python training/scripts/dump_chat_template.py --model training/outputs/merged
# -> outputs/chat_template_render.json: the golden prompt cases rendered by the real template.
#    The iOS tests can compare llama.cpp's rendering of the same messages against it.
```

## 6. Re-evaluate the quantised model, then ship

```bash
llama-server -m training/outputs/gguf/bible-gemma-q4_k_m.gguf -c 6144 --port 8080 &
python training/scripts/evaluate.py --backend openai --base-url http://localhost:8080/v1 --label q4_k_m
python training/scripts/compare_reports.py base=training/outputs/eval/base \
  finetuned=training/outputs/eval/finetuned q4_k_m=training/outputs/eval/q4_k_m \
  --out training/outputs/eval/comparison.md
```

**Ship gate** (suggested):

- The fine-tuned model beats the base model on correct refusals, unsupported citations and format adherence, with no rise in false refusals.
- Q4_K_M stays within about 3 points of the fine-tuned model on every metric.
- Citation validity is ≥ 0.98.

If all three hold, upload the GGUF and fill in `sha256`/`bytes` in the model manifest.

The memory and summary rows in the comparison table guard the auxiliary tasks. Memory JSON validity
should be ≥ 0.98, and the fine-tuned model must not fall below the base model on memory type/key
match, `empty_accuracy` or the summary word limit. If they regress, add more auxiliary seeds. The
held-out auxiliary test set is small: 2 memory and 1 summary record with the default split.
