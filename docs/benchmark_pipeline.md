## Vision-Language Benchmark Pipeline (updated Dec 28, 2025)

### What you get
- Backend choice (vLLM, TensorRT-LLM, Hugging Face TGI) per model for best speed vs. effort.
- One async client script `benchmark.py` that hits any OpenAI-compatible endpoint and writes a CSV.
- Ready-to-run serve scripts in `scripts/` for each backend.

### Backend recommendations (fastest first when supported)
| Model | Recommended backend | Notes |
| --- | --- | --- |
| Qwen/Qwen2.5-VL-7B-Instruct | TensorRT-LLM → vLLM | TRT-LLM 0.20.0 adds Qwen2.5-VL vision support; vLLM works if engines unavailable. citeturn2search0 |
| Qwen/Qwen2.5-VL-3B-Instruct | TensorRT-LLM → vLLM | Same as above; 3B lighter. citeturn2search0 |
| Qwen/Qwen3-VL-8B-Instruct | vLLM | vLLM recipe documents Qwen3-VL serving. citeturn0search4 |
| Qwen/Qwen3-VL-4B-Instruct | vLLM | As above. citeturn0search4 |
| InternVL3.5-1B/2B/4B-Instruct | vLLM → TGI | HF card notes vLLM works for InternVL3.5. citeturn3search8 |
| erax-ai/EraX-VL-2B-V1.5 | TGI (transformers) | Community HF model; run with `--trust-remote-code` in TGI. |
| 5CD-AI/Vintern-1B-v3_5 | TGI (transformers) | Community HF model; use TGI transformers path. |

### Why these picks
- **vLLM**: Native multimodal path for Qwen3-VL and works for InternVL3.5 (official recipe). citeturn2search0turn1search1
- **TensorRT-LLM**: Support matrix lists `Qwen/Qwen2.5-VL-7B-Instruct` with image+video; fastest once engines are built. citeturn3search0
- **TGI**: Vision-language inference mode exists for multimodal models. citeturn1search0

### GPU VRAM guidelines (single-GPU FP16/BF16 unless noted)
| Model | Minimum (runs) | Comfortable | Notes |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B | 8–10 GB (INT4) / 12 GB (BF16) | 16 GB | BF16 params ≈5.7 GB; allow overhead. citeturn0search0 |
| Qwen2.5-VL-7B | 16 GB (BF16) | 24 GB+ | BF16 params ≈13 GB; typical usage 14–16 GB. citeturn0search0turn3search4 |
| Qwen3-VL-4B | 12 GB | 16 GB | FP16 guide shows 10–12 GB. citeturn1search6 |
| Qwen3-VL-8B | 20 GB | 24 GB+ | HF card recommends 20 GB FP16. citeturn0search3 |
| InternVL3.5-1B/2B/4B | 12 GB | 16–24 GB | Up to 30B fits single A100; small variants fine on 16 GB. citeturn1search1 |
| EraX-VL-2B-V1.5 | 8 GB (INT4) / 12 GB (FP16) | 16 GB | Similar size to InternVL 2B. |
| Vintern-1B-v3_5 | 8 GB | 12 GB | Small VLM; INT4 lowers further. |

Multi-GPU: both vLLM and TensorRT-LLM support tensor parallelism; split models when VRAM is tight (e.g., TP=2 across two 12 GB GPUs for Qwen2.5-VL-7B).

### Prereqs on the remote GPU host
- CUDA 12.x + recent NVIDIA driver.
- Python 3.10+ (for the benchmark client).
- Docker with GPU runtime (for serving containers).
- Disk cache for HF weights (default `~/.cache/huggingface`).

### Serve the models
- vLLM (default port 8000):
  - `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct PORT=8000 ./scripts/serve_vllm.sh`
- Hugging Face TGI (default port 8080):
  - `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct PORT=8080 ./scripts/serve_tgi.sh`
  - Add `-e HF_TOKEN=...` if the model is gated.
- TensorRT-LLM (default port 9000) for Qwen2.5-VL:
  - `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct PORT=9000 ./scripts/serve_trtllm.sh`
  - For production, pre-build engines with `trtllm-build` matching GPU/TP size.

All three scripts expose an OpenAI-compatible endpoint at `http://localhost:<port>/v1`.

Fast client one-liner:
```bash
DATA=data/images MODEL="Qwen/Qwen3-VL-8B-Instruct" BACKEND=vllm ./scripts/bench.sh
```

### Prepare your dataset
- Directory of images: `data/images/*.{jpg,png}` uses a single prompt.
- Or JSONL with per-sample prompts:
```json
{"image": "data/images/cat.jpg", "prompt": "Describe the scene."}
{"image": "data/images/diagram.png", "prompt": "Explain the chart."}
```

### Run the benchmark (now more robust)
Install client deps once:
```bash
pip install -r requirements.txt
```

Example run against vLLM:
```bash
python benchmark.py \
  --data data/images \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --prompt "Give a concise caption." \
  --concurrency 8 \
  --retries 1 \
  --request-timeout 60 \
  --max-side 1024 \
  --shuffle --samples 200 \
  --output qwen3vl8b_vllm.csv
```

Hit TensorRT-LLM:
```bash
python benchmark.py \
  --data dataset.jsonl \
  --endpoint http://localhost:9000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --backend trtllm \
  --retries 2 \
  --max-side 960 \
  --output qwen25vl_trt.csv
```

CSV columns: `image,prompt,latency_ms,finish_reason,output,error,backend,model`

Client resiliency:
- Endpoint preflight: lists `/v1/models` and warns if the target model ID isn’t exposed.
- Per-request timeout (`--request-timeout`) plus retry count (`--retries`).
- Optional client-side resize (`--max-side`) to shrink upload bandwidth.
- Shuffle/limit (`--shuffle --samples N`) for quick spot checks.

### Performance knobs (quick wins)
- vLLM: set `TP` to number of GPUs; use `--dtype bfloat16` (A100/H100) or `--quantization awq/int4` if available for memory saving.
- TensorRT-LLM: build FP8 or INT8 KV+GEMM engines for speed; match `tensor-parallel-size` to GPU count.
- TGI: bump `MAX_BATCH_PREFILL_TOKENS` / `MAX_TOTAL_TOKENS` envs for higher throughput; ensure enough `--num-shard` to fill GPUs.
- Client: increase `--concurrency` until latency stops improving; optionally resize images with `--max-side` to reduce upload time.

### Verification checklist before a run
- GPU memory headroom (watch `nvidia-smi`).
- Endpoint reachable: `curl http://localhost:8000/v1/models`.
- Sanity request returns text (one sample image).

### Extending
- Add new models by pointing `MODEL_ID` to HF repo (TGI/vLLM) or TRT-LLM config.
- Swap prompts per sample by switching to JSONL input.
- Attach accuracy metrics by post-processing `output` vs references in the CSV.
