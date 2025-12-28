#!/usr/bin/env bash
set -euo pipefail

# Launch Hugging Face Text Generation Inference with multimodal support.
# Tested with ghcr.io/huggingface/text-generation-inference:3.3.5 (requires CUDA 12).

MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-VL-7B-Instruct"}
PORT=${PORT:-8080}
NUM_SHARD=${NUM_SHARD:-1}
CONTAINER=${CONTAINER:-"ghcr.io/huggingface/text-generation-inference:3.3.5"}

docker run --gpus all --rm --net host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MAX_BATCH_PREFILL_TOKENS=32768 \
  -e MAX_INPUT_LENGTH=4096 \
  -e MAX_TOTAL_TOKENS=6144 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/data \
  "${CONTAINER}" \
  --model-id "${MODEL_ID}" \
  --port "${PORT}" \
  --hostname 0.0.0.0 \
  --num-shard "${NUM_SHARD}" \
  --dtype bfloat16 \
  --enable-cors

# After start, the OpenAI-compatible endpoint is http://localhost:${PORT}/v1
