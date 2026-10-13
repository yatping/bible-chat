#!/usr/bin/env bash
# Export google/embeddinggemma-300m to GGUF (f16, then Q8_0) for on-device query embedding.
#
# Requirements: git, cmake, a C/C++ toolchain, python3, and Hugging Face access to the gated model
# (accept the licence on huggingface.co, then `hf auth login` or export HF_TOKEN).
#
# Usage: content/scripts/export_embedder_gguf.sh [OUT_DIR]          (default: content/build/gguf)
# Env:   LLAMA_CPP_REF   llama.cpp git tag/commit to pin (default below)
#        VERIFY=1        also build llama-embedding and compare GGUF vs sentence-transformers vectors
#
# IMPORTANT: sentence-transformers' EmbeddingGemma = mean pooling + two Dense projections. The GGUF
# must include those Dense layers (--sentence-transformers-dense-modules), otherwise on-device query
# vectors will not live in the same space as the document vectors built by build_content.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTENT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${1:-$CONTENT_DIR/build/gguf}"
WORK_DIR="$CONTENT_DIR/build/llama.cpp-work"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b11500}"
MODEL_ID="google/embeddinggemma-300m"
PY="${PYTHON:-$CONTENT_DIR/.venv/bin/python}"

mkdir -p "$OUT_DIR" "$WORK_DIR"

if [ ! -d "$WORK_DIR/llama.cpp/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$WORK_DIR/llama.cpp"
fi
git -C "$WORK_DIR/llama.cpp" fetch --tags --quiet origin
git -C "$WORK_DIR/llama.cpp" checkout --quiet "$LLAMA_CPP_REF"
echo "llama.cpp pinned at $(git -C "$WORK_DIR/llama.cpp" rev-parse HEAD) ($LLAMA_CPP_REF)"

"$PY" -m pip install --quiet -r "$WORK_DIR/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub

MODEL_DIR="$WORK_DIR/embeddinggemma-300m"
"$PY" - "$MODEL_ID" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
EOF

CONVERT="$WORK_DIR/llama.cpp/convert_hf_to_gguf.py"
CONVERT_HELP="$("$PY" "$CONVERT" --help)"
if [[ "$CONVERT_HELP" != *"--sentence-transformers-dense-modules"* ]]; then
  echo "error: llama.cpp $LLAMA_CPP_REF lacks --sentence-transformers-dense-modules; use a newer LLAMA_CPP_REF" >&2
  exit 1
fi

F16="$OUT_DIR/embeddinggemma-300m-f16.gguf"
Q8="$OUT_DIR/embeddinggemma-300m-Q8_0.gguf"
"$PY" "$CONVERT" "$MODEL_DIR" --outtype f16 --sentence-transformers-dense-modules --outfile "$F16"

cmake -S "$WORK_DIR/llama.cpp" -B "$WORK_DIR/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WORK_DIR/build" --config Release -j --target llama-quantize llama-embedding >/dev/null
"$WORK_DIR/build/bin/llama-quantize" "$F16" "$Q8" Q8_0

if [ "${VERIFY:-0}" = "1" ]; then
  # Compare GGUF (llama.cpp, mean pooling from GGUF metadata) with sentence-transformers.
  TEXT="task: search result | query: Why did Jesus tell Peter to forgive seventy times seven?"
  "$WORK_DIR/build/bin/llama-embedding" -m "$Q8" --embd-normalize 2 --embd-output-format json \
    -p "$TEXT" 2>/dev/null > "$OUT_DIR/verify_gguf.json"
  "$PY" - "$MODEL_DIR" "$TEXT" "$OUT_DIR/verify_gguf.json" <<'EOF'
import json, math, sys
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(sys.argv[1])
ref = model.encode([sys.argv[2]], prompt="")[0][:256]
gguf = json.load(open(sys.argv[3]))["data"][0]["embedding"][:256]
def unit(v):
    n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]
cos = sum(a * b for a, b in zip(unit(list(map(float, ref))), unit(gguf)))
print(f"cosine(sentence-transformers, GGUF Q8_0) on first 256 dims = {cos:.4f} (expect > 0.99)")
sys.exit(0 if cos > 0.99 else 1)
EOF
fi

echo
echo "GGUF files:"
for f in "$F16" "$Q8"; do
  printf '  %s  %s bytes\n' "$f" "$(wc -c < "$f" | tr -d ' ')"
  shasum -a 256 "$f" | awk '{print "  sha256 " $1}'
done
