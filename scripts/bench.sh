#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper to minimize typing when running the benchmark.
# Example:
#   DATA=data/images MODEL="Qwen/Qwen3-VL-8B-Instruct" BACKEND=vllm ./scripts/bench.sh

DATA=${DATA:?set DATA to an image dir or jsonl}
MODEL=${MODEL:?set MODEL to HF id as seen by the server}
ENDPOINT=${ENDPOINT:-http://localhost:8000/v1}
BACKEND=${BACKEND:-vllm}
PROMPT=${PROMPT:-"Describe the image."}
OUT=${OUT:-results.csv}
CONCURRENCY=${CONCURRENCY:-8}
MAX_SIDE=${MAX_SIDE:-1024}
RETRIES=${RETRIES:-1}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-60}
SAMPLES=${SAMPLES:-}
SHUFFLE=${SHUFFLE:-1} # set to 0 to keep order

ARGS=(
  --data "$DATA"
  --endpoint "$ENDPOINT"
  --model "$MODEL"
  --backend "$BACKEND"
  --prompt "$PROMPT"
  --concurrency "$CONCURRENCY"
  --max-side "$MAX_SIDE"
  --retries "$RETRIES"
  --request-timeout "$REQUEST_TIMEOUT"
  --output "$OUT"
)

if [[ -n "$SAMPLES" ]]; then
  ARGS+=(--samples "$SAMPLES")
fi

if [[ "${SHUFFLE}" != "0" ]]; then
  ARGS+=(--shuffle)
fi

python benchmark.py "${ARGS[@]}"
