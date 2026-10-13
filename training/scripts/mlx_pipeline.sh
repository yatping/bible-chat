#!/usr/bin/env bash
# QLoRA fine-tuning of Gemma 4 E2B on Apple Silicon with mlx-lm, end to end:
#
#   dataset → split → mlx data → 4-bit base → QLoRA → evaluate base vs tuned → merge into HF checkpoint → GGUF Q4_K_M
#
# Usage: training/scripts/mlx_pipeline.sh [step...]
#   steps: data quantize train eval merge gguf   (default: all, in order)
# Env:   PY (training venv python, default training/.venv-mlx/bin/python), BASE (default google/gemma-4-E2B-it),
#        LIMIT (optional: evaluate only the first N test records)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-training/.venv-mlx/bin/python}"
BIN="$(dirname "$PY")"
BASE="${BASE:-google/gemma-4-E2B-it}"
OUT=training/outputs
Q4="$OUT/mlx/gemma-4-E2B-it-4bit"
ADAPTER="$OUT/mlx/adapter"
MERGED="$OUT/bible-gemma-merged"
if (($#)); then STEPS=("$@"); else STEPS=(data quantize train eval merge gguf); fi
EVAL_LIMIT=""
[[ -n "${LIMIT:-}" ]] && EVAL_LIMIT="--limit $LIMIT"

step() { echo; echo "==> $*"; }

for s in "${STEPS[@]}"; do
  case "$s" in
    data)
      step "Build, split and convert the dataset"
      "$PY" training/scripts/build_dataset.py --distractors-per-record 3
      "$PY" training/scripts/split_dataset.py
      "$PY" training/scripts/prepare_mlx_data.py
      ;;
    quantize)
      step "Quantize $BASE to 4-bit MLX (QLoRA base)"
      [[ -d "$Q4" ]] || "$BIN/mlx_lm.convert" --hf-path "$BASE" --mlx-path "$Q4" -q --q-bits 4 --q-group-size 64
      ;;
    train)
      step "QLoRA (configs/mlx_qlora.yaml)"
      "$PY" training/scripts/train_mlx.py -c training/configs/mlx_qlora.yaml
      ;;
    eval)
      step "Evaluate base vs fine-tuned on the held-out test split (greedy, same 4-bit base)"
      "$PY" training/scripts/evaluate.py --backend mlx --model "$Q4" --label mlx-base --greedy $EVAL_LIMIT
      "$PY" training/scripts/evaluate.py --backend mlx --model "$Q4" --adapter "$ADAPTER" --label mlx-finetuned \
        --greedy $EVAL_LIMIT
      "$PY" training/scripts/compare_reports.py base="$OUT/eval/mlx-base" finetuned="$OUT/eval/mlx-finetuned" --out "$OUT/eval/comparison.md"
      ;;
    merge)
      step "Merge the adapter into the full-precision HF checkpoint"
      "$PY" training/scripts/merge_mlx_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"
      ;;
    gguf)
      step "Convert to GGUF and quantize to Q4_K_M (llama.cpp)"
      training/scripts/convert_gguf.sh "$MERGED"
      ;;
    *) echo "unknown step: $s" >&2; exit 2 ;;
  esac
done
