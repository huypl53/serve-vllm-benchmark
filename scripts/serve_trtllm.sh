#!/usr/bin/env bash
set -euo pipefail

# Minimal TensorRT-LLM serve helper for Qwen2.5-VL on NVIDIA GPUs.
# Uses PyTorch fallback path (no prebuilt engine required) so you can prototype quickly.
# For production, build TensorRT engines with trtllm-build for the target GPU.

MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-VL-7B-Instruct"}
PORT=${PORT:-9000}
TP=${TP:-1}
CONTAINER=${CONTAINER:-"nvcr.io/nvidia/tensorrtllm/tensorrtllm:25.06-py3"}

docker run --gpus all --rm --net host \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  "${CONTAINER}" \
  trtllm-serve \
  --model "${MODEL_ID}" \
  --tensor-parallel-size "${TP}" \
  --enable-multi-modal \
  --enable-openai-api \
  --openai-api-port "${PORT}" \
  --dtype bfloat16

# Endpoint will be http://localhost:${PORT}/v1
