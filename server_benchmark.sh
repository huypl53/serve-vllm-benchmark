#!/usr/bin/env bash
set -euo pipefail

# Server-side one-shot benchmark runner.

REPO_URL="${REPO_URL:-https://github.com/huypl53/serve-vllm-benchmark.git}"
REPO_DIR="${REPO_DIR:-serve-vllm-benchmark}"
BRANCH="${BRANCH:-feat/benchmark-v2}"
IMAGE_URL="${IMAGE_URL:-https://pub-d34d24b39e444483bea64e0a75ccb2b8.r2.dev/test_img.zip}"

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Install uv if missing.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install dependencies via uv.
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download and extract images.
mkdir -p images
tmp_zip="$(mktemp)"
curl -L "${IMAGE_URL}" -o "${tmp_zip}"
unzip -o "${tmp_zip}" -d images
rm -f "${tmp_zip}"

# Run benchmark.
./run_all.sh --images ./images
