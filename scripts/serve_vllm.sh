#!/usr/bin/env bash
set -euo pipefail

# Simple helper to launch a multimodal model with vLLM's OpenAI-compatible server.
# Requirements: vLLM >= 0.11.0, CUDA GPU. Install via pip or use the official Docker image.

MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-VL-7B-Instruct"}
PORT=${PORT:-8000}
TP=${TP:-1}
DTYPE=${DTYPE:-"bfloat16"}  # use fp16 if GPU lacks bfloat16
HOST=${HOST:-"0.0.0.0"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

echo "Starting vLLM for ${MODEL_ID} on port ${PORT} (TP=${TP}, dtype=${DTYPE})"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --disable-logs 0 \
  --trust-remote-code
