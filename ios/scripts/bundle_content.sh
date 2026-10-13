#!/usr/bin/env bash
# Copies the content pipeline output (content/build/) into the app's bundled resources.
# Usage: ios/scripts/bundle_content.sh [path/to/content/build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${1:-${SCRIPT_DIR}/../../content/build}"
DEST="${SCRIPT_DIR}/../BibleAI/Resources/Content"

for f in bible.sqlite rag_embeddings.f16; do
  if [[ ! -f "${SOURCE}/${f}" ]]; then
    echo "error: ${SOURCE}/${f} not found – build the content first (see content/README)." >&2
    exit 1
  fi
done

# Sanity check: the embedding file size must match content_meta (count × dim × 2 bytes).
read -r COUNT DIM < <(sqlite3 "${SOURCE}/bible.sqlite" \
  "SELECT (SELECT value FROM content_meta WHERE key='embedding_count'), (SELECT value FROM content_meta WHERE key='embedding_dim');" | tr '|' ' ')
EXPECTED=$((COUNT * DIM * 2))
ACTUAL=$(stat -f%z "${SOURCE}/rag_embeddings.f16")
if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
  echo "error: rag_embeddings.f16 is ${ACTUAL} bytes, content_meta says ${COUNT}×${DIM}×2 = ${EXPECTED}" >&2
  exit 1
fi

mkdir -p "${DEST}"
cp "${SOURCE}/bible.sqlite" "${SOURCE}/rag_embeddings.f16" "${DEST}/"
echo "Bundled ${COUNT} chunks (dim ${DIM}) into ${DEST}"
