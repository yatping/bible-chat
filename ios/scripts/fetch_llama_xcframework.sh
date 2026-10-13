#!/usr/bin/env bash
# Downloads the pinned llama.cpp XCFramework release into ios/Frameworks/llama.xcframework.
#
# The release asset is published by llama.cpp's CI on every build tag:
#   https://github.com/ggml-org/llama.cpp/releases/tag/${LLAMA_TAG}
# To upgrade: change LLAMA_TAG and LLAMA_SHA256 (the asset "digest" from
#   curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/<tag>
# ) and LLAMA_COMMIT (`git ls-remote https://github.com/ggml-org/llama.cpp refs/tags/<tag>`), re-run this
# script with --force and rebuild. Check GemmaEngine / LlamaEmbeddingEngine still compile: the C API changes often.
#
# Simulator slice: upstream release CI only builds `macos` + `ios-device`
# (see .github/workflows/release.yml, "only macos and ios-device due to long build time").
# To run the app in the iOS Simulator with the real engine linked, this script clones the same
# tag (commit verified), builds the `ios-sim` slice with upstream's build-xcframework.sh and merges
# it into the downloaded XCFramework. Requires cmake (brew install cmake). Takes a few minutes.
# Pass --no-simulator to skip (device-only framework; simulator builds then fall back to the mock engine
# only if you also remove the framework from the target).
#
# usage: fetch_llama_xcframework.sh [--force] [--no-simulator]
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b11580}"
LLAMA_SHA256="${LLAMA_SHA256:-d948d89e27687cd7aef9275fdc365a62855f93387a57f732959c9e40171a8152}"
LLAMA_COMMIT="${LLAMA_COMMIT:-4413a192410fad1f054e9216114087a3e01527a0}"

FORCE=0
WITH_SIMULATOR=1
for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    --no-simulator) WITH_SIMULATOR=0 ;;
    *) echo "unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/../Frameworks"
ASSET="llama-${LLAMA_TAG}-xcframework.zip"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ASSET}"
STAMP="${DEST_DIR}/llama.xcframework/.release-tag"
STAMP_VALUE="${LLAMA_TAG}$([[ ${WITH_SIMULATOR} == 1 ]] && echo '+sim')"

if [[ -f "${STAMP}" && "$(cat "${STAMP}")" == "${STAMP_VALUE}" && ${FORCE} == 0 ]]; then
  echo "llama.xcframework ${LLAMA_TAG} already present (use --force to re-download)."
  exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "Downloading ${URL}"
curl --fail --location --retry 3 --progress-bar -o "${WORK_DIR}/${ASSET}" "${URL}"

ACTUAL_SHA256="$(shasum -a 256 "${WORK_DIR}/${ASSET}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${LLAMA_SHA256}" ]]; then
  echo "error: sha256 mismatch for ${ASSET}" >&2
  echo "  expected ${LLAMA_SHA256}" >&2
  echo "  actual   ${ACTUAL_SHA256}" >&2
  exit 1
fi
echo "sha256 verified."

mkdir -p "${WORK_DIR}/unzipped"
unzip -q "${WORK_DIR}/${ASSET}" -d "${WORK_DIR}/unzipped"
FRAMEWORK="$(find "${WORK_DIR}/unzipped" -maxdepth 3 -type d -name 'llama.xcframework' | head -n 1)"
if [[ -z "${FRAMEWORK}" ]]; then
  echo "error: llama.xcframework not found inside ${ASSET}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
rm -rf "${DEST_DIR}/llama.xcframework"

if [[ ${WITH_SIMULATOR} == 1 ]]; then
  command -v cmake >/dev/null || { echo "error: cmake is required for the simulator slice (or pass --no-simulator)" >&2; exit 1; }
  echo "Building iOS Simulator slice from source (${LLAMA_TAG} @ ${LLAMA_COMMIT})..."
  git clone --quiet --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp.git "${WORK_DIR}/src"
  HEAD_COMMIT="$(git -C "${WORK_DIR}/src" rev-parse HEAD)"
  if [[ "${HEAD_COMMIT}" != "${LLAMA_COMMIT}" ]]; then
    echo "error: tag ${LLAMA_TAG} resolves to ${HEAD_COMMIT}, expected ${LLAMA_COMMIT}" >&2
    exit 1
  fi
  (cd "${WORK_DIR}/src" && ./build-xcframework.sh ios-sim > "${WORK_DIR}/sim-build.log" 2>&1) || {
    tail -50 "${WORK_DIR}/sim-build.log" >&2; exit 1; }
  SIM_DIR="${WORK_DIR}/src/build-ios-sim"
  ARGS=()
  for slice in "${FRAMEWORK}"/*/; do
    slice="${slice%/}"
    ARGS+=(-framework "${slice}/llama.framework")
    [[ -d "${slice}/dSYMs/llama.dSYM" ]] && ARGS+=(-debug-symbols "${slice}/dSYMs/llama.dSYM")
  done
  ARGS+=(-framework "${SIM_DIR}/framework/llama.framework")
  [[ -d "${SIM_DIR}/dSYMs/llama.dSYM" ]] && ARGS+=(-debug-symbols "${SIM_DIR}/dSYMs/llama.dSYM")
  xcrun xcodebuild -create-xcframework "${ARGS[@]}" -output "${DEST_DIR}/llama.xcframework" >/dev/null
else
  cp -R "${FRAMEWORK}" "${DEST_DIR}/llama.xcframework"
fi
echo "${STAMP_VALUE}" > "${STAMP}"
echo "Installed ${DEST_DIR}/llama.xcframework (${LLAMA_TAG})."
