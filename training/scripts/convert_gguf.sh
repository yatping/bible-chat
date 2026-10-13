#!/usr/bin/env bash
# Convert the merged HF model to GGUF f16, quantise to Q4_K_M, print size + sha256.
#
# Usage:
#   training/scripts/convert_gguf.sh [MERGED_DIR] [OUT_DIR]
#
# Environment:
#   LLAMA_CPP_DIR   where to clone/build llama.cpp     (default: training/outputs/llama.cpp)
#   LLAMA_CPP_REF   git tag/commit to pin               (default: b11500, same tag as
#                   content/scripts/export_embedder_gguf.sh; keep in sync with the iOS llama.cpp build)
#   OUT_NAME        output file stem                    (default: bible-gemma)
#   QUANT           quantisation type                   (default: Q4_K_M)
#   SMOKE=1         run a short llama-cli prompt against the quantised model
#   PYTHON          python with the conversion requirements (default: python3)
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$TRAINING_DIR")"

MERGED_DIR="${1:-$TRAINING_DIR/outputs/merged}"
OUT_DIR="${2:-$TRAINING_DIR/outputs/gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$TRAINING_DIR/outputs/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b11500}"
OUT_NAME="${OUT_NAME:-bible-gemma}"
QUANT="${QUANT:-Q4_K_M}"
PYTHON="${PYTHON:-python3}"

[[ -f "$MERGED_DIR/config.json" ]] || { echo "no merged model at $MERGED_DIR (run merge_adapter.py)" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# 1. Clone + pin llama.cpp
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --tags --quiet origin
git -C "$LLAMA_CPP_DIR" checkout --quiet "$LLAMA_CPP_REF"
if [[ "$LLAMA_CPP_REF" == "master" ]]; then git -C "$LLAMA_CPP_DIR" pull --quiet --ff-only origin master; fi
LLAMA_COMMIT="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "llama.cpp @ $LLAMA_CPP_REF ($LLAMA_COMMIT)"

# 2. Build the tools we need
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j --target llama-quantize llama-cli >/dev/null
BIN="$LLAMA_CPP_DIR/build/bin"

# 3. HF -> GGUF f16 (the processor's chat template is embedded in the GGUF metadata)
"$PYTHON" -m pip install -q -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
F16="$OUT_DIR/$OUT_NAME-f16.gguf"
"$PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outtype f16 --outfile "$F16"

# 4. Quantise
QOUT="$OUT_DIR/$OUT_NAME-$(echo "$QUANT" | tr '[:upper:]' '[:lower:]').gguf"
"$BIN/llama-quantize" "$F16" "$QOUT" "$QUANT"

# 5. Size + sha256 (fill these into the model manifest)
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
bytes() { wc -c <"$1" | tr -d ' '; }
for f in "$F16" "$QOUT"; do
  echo "$(basename "$f")  bytes=$(bytes "$f")  sha256=$(sha "$f")"
done
cat >"$OUT_DIR/$OUT_NAME-$QUANT.manifest.json" <<EOF
{"filename": "$(basename "$QOUT")", "bytes": $(bytes "$QOUT"), "sha256": "$(sha "$QOUT")", "quant": "$QUANT", "llama_cpp_commit": "$LLAMA_COMMIT"}
EOF

# 6. Optional smoke test (flags verified against llama.cpp b11500 tools/cli: -sys, -st/--single-turn,
#    -n, -c, --temp, --top-p, --top-k, --repeat-penalty)
if [[ "${SMOKE:-0}" == "1" ]]; then
  SYSTEM_PROMPT="$(cat "$REPO_ROOT/shared/system_prompt.txt")"
  USER_PROMPT=$'SOURCES\n[S1] type: scripture | reference: John 11:35\nJesus wept.\n\nQUESTION\nWhat did Jesus do in John 11:35?'
  "$BIN/llama-cli" -m "$QOUT" -sys "$SYSTEM_PROMPT" -p "$USER_PROMPT" -n 128 -c 6144 \
    --temp 0.3 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 -st
fi
echo "done: $QOUT"
