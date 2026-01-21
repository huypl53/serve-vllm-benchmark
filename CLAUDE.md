# VLM Benchmark - Developer Guide

This guide documents internal knowledge about the benchmark-v2 codebase for contributors.

## Running Code

To run a Python program with uv:
```bash
uv run <path/to/script.py>
```

For shell scripts on remote servers, optimize for speed and minimal dependencies.

---

## Table of Contents

1. [Adding a New Model](#adding-a-new-model)
2. [Adding a New Platform](#adding-a-new-platform)
3. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
4. [Key Files Reference](#key-files-reference)

---

## Adding a New Model

### Step 1: Add Model Configuration

**File:** `config/models.yaml`

Add a new entry under the `models:` key:

```yaml
models:
  my-new-model:
    huggingface_id: "organization/Model-Name"
    display_name: "My Model Display Name"
    memory_estimate_gb: 16      # Approximate VRAM needed
    min_vram_gb: 14             # Minimum VRAM to run
    quantization: null          # Or "w8a8", "w4a16" for quantized models
    platforms:
      vllm:
        supported: true
        args:
          trust_remote_code: true
          max_model_len: 8192
          gpu_memory_utilization: 0.9
          limit_mm_per_prompt: 1
      tgi:
        supported: true
        args:
          max_input_length: 8192
          max_total_tokens: 16384
      tensorrt:
        supported: true
        backend: pytorch
        args: {}
```

**Important Config Values:**
- `max_model_len` (vLLM): Maximum context length including prompt + generated tokens
- `max_input_length` (TGI): Maximum prompt tokens
- `max_total_tokens` (TGI): Prompt + generated tokens combined
- `limit_mm_per_prompt` (vLLM): Number of images per request (usually 1 for VLMs)

### Step 2: Update GPU Profiles (Optional)

**File:** `config/models.yaml` (under `gpu_profiles:`)

Add the model to appropriate GPU profiles if you want it recommended for certain GPUs:

```yaml
gpu_profiles:
  T4:
    supported_models:
      - my-new-model  # Add here if it runs on 16GB
```

### Step 3: Update Documentation (If releasing)

**File:** `docs/SETUP.md`

Add the model to the "Supported Models" table:

```markdown
| Model | vLLM | TGI | TensorRT | VRAM |
|-------|------|-----|----------|------|
| My Model | Yes | Yes | No | ~16GB |
```

### Step 4: Test the Model

```bash
# List to verify config loads
uv run python -m src.cli list

# Run a quick benchmark
./run_all.sh --models my-new-model --max-images 5
```

---

## Adding a New Platform

### Architecture Overview

The platform system uses an **abstract base class** pattern:

```
src/platforms/base.py          # Abstract BasePlatform class
src/platforms/vllm_platform.py # vLLM implementation
src/platforms/tgi_platform.py  # TGI implementation
src/platforms/tensorrt_platform.py # TensorRT implementation
```

### Step 1: Create Platform Implementation

**File:** `src/platforms/<platform>_platform.py`

Create a new class extending `BasePlatform`:

```python
from .base import BasePlatform, InferenceResult, ModelInfo

class MyPlatform(BasePlatform):
    @property
    def platform_name(self) -> str:
        return "myplatform"  # Used in CLI, configs, results

    def _do_connect(self, server_url: str) -> bool:
        # Initialize your client here
        # Return True on success
        pass

    def _do_health_check(self) -> bool:
        # Check if server is responding
        pass

    def _do_inference(self, image, prompt, max_new_tokens, temperature, iteration=0, **kwargs) -> InferenceResult:
        # Run single inference and return InferenceResult
        pass

    def _do_inference_async(self, image, prompt, max_new_tokens, temperature, iteration=0, **kwargs) -> InferenceResult:
        # (Optional) Implement true async for concurrent requests
        # Default: BasePlatform provides executor-based fallback
        pass

    def _do_get_model_info(self) -> ModelInfo:
        # Return model metadata from server
        pass
```

### Step 2: Register Platform

**File:** `src/platforms/__init__.py`

```python
from .myplatform_platform import MyPlatform

PLATFORM_REGISTRY = {
    "vllm": VLLMPlatform,
    "tgi": TGIPlatform,
    "tensorrt": TensorRTPlatform,
    "myplatform": MyPlatform,  # Add here
}
```

### Step 3: Update Model Configs

**File:** `config/models.yaml`

Add your platform to all applicable models:

```yaml
models:
  qwen2.5-vl-7b:
    platforms:
      myplatform:
        supported: true
        args:
          # Platform-specific args
```

### Step 4: Add Docker Compose File (Optional)

**File:** `docker/myplatform.yaml`

```yaml
services:
  myplatform-server:
    image: myplatform-image:latest
    container_name: vlm-benchmark-myplatform
    environment:
      - MODEL_ID=${MODEL_ID}
    ports:
      - "${MYPLATFORM_PORT:-8002}:8000"
    # ... GPU configuration, health checks, etc.
```

### Step 5: Update run_all.sh

**File:** `run_all.sh`

Add your platform to:
- Default `--all-platforms` list
- Port variable (`MYPLATFORM_PORT=8002`)
- `start_server()` function
- `stop_server()` function
- Server URL mapping

### Step 6: Update Server Port Config

**File:** `config/benchmark.yaml`

```yaml
server:
  ports:
    vllm: 8000
    tgi: 8080
    tensorrt: 8001
    myplatform: 8002  # Add here
```

---

## Common Pitfalls to Avoid

### 1. Model Length Limits

**Problem:** `The decoder prompt (length XXXX) is longer than the maximum model length of 4096.`

**Root Cause:** The model's `max_model_len` is too small for the image + prompt combination.

**Locations to Update:**
- `config/models.yaml` - Set `max_model_len: 8192` or higher
- `docker/vllm.yaml` - Update `--max-model-len ${MAX_MODEL_LEN:-8192}`
- `docker/tgi.yaml` - Update `--max-input-length` and `--max-total-tokens`
- `src/platforms/base.py` - Update `max_model_len` default in `ModelInfo`

### 2. Image Caching Skewing Results

**Problem:** Second and third iterations show unrealistically fast TTFT because vLLM caches the image encoding.

**Solution:** The `_bypass_image_cache()` method in `BasePlatform` tweaks a single pixel by ±1 RGB value for each iteration after the first. This makes each request unique while not affecting model output.

**Usage:** Always pass `iteration=` parameter to `_do_inference()`:
```python
self._do_inference(image, prompt, ..., iteration=iteration)
```

### 3. Concurrent Request OOM

**Problem:** Using high concurrency causes GPU OOM errors.

**Mitigation:**
- Start with `concurrency=2` and gradually increase
- Monitor VRAM usage during benchmarks
- Reduce `gpu_memory_utilization` in docker configs
- Use smaller models or quantization

### 4. Platform Not Actually Supported

**Problem:** Marking a model as `supported: true` when the platform doesn't actually support that model architecture.

**Example:** TGI doesn't support Qwen3-VL architecture (as of Dec 2025).

**Solution:** Test before enabling. Check:
- Platform documentation
- Platform GitHub issues
- Actual inference with the model

### 5. Forgetting to Update run_all.sh

**Problem:** Adding model to `config/models.yaml` but forgetting to add to `--all-models` list in `run_all.sh`.

**Result:** Model won't be included in full benchmarks.

**Solution:** Always update the `MODELS` list in `--all-models)` case.

### 6. Docker Container Model Download Time

**Problem:** First benchmark includes model download time, skewing results.

**Solution:** Models are cached to `${HF_CACHE:-~/.cache/huggingface}` volume. Pre-pull models:
```bash
MODEL_ID=org/model docker compose -f docker/vllm.yaml up
docker logs -f vlm-benchmark-vllm  # Wait for download
docker compose -f docker/vllm.yaml down
```

### 7. Gated Models Without HF_TOKEN

**Problem:** Gated models (Llama, etc.) fail with authentication errors.

**Solution:**
```bash
export HF_TOKEN=your_token_here
# Or login: huggingface-cli login
# Accept license on HuggingFace website first
```

---

## Key Files Reference

### Configuration Files

| File | Purpose |
|------|---------|
| `config/models.yaml` | Model definitions, platform support, GPU profiles |
| `config/benchmark.yaml` | Benchmark settings (iterations, tokens, concurrency) |

### Source Code

| File | Purpose |
|------|---------|
| `src/cli.py` | CLI entry point, argument parsing, command dispatch |
| `src/benchmarker.py` | Benchmark orchestration, metrics aggregation |
| `src/config.py` | Config loading, dataclasses for settings |
| `src/platforms/base.py` | Abstract `BasePlatform`, `InferenceResult`, retry logic |
| `src/platforms/vllm_platform.py` | vLLM implementation (OpenAI API) |
| `src/platforms/tgi_platform.py` | TGI implementation (HTTP API) |
| `src/platforms/tensorrt_platform.py` | TensorRT implementation |
| `src/data/image_loader.py` | Image discovery and validation |
| `src/metrics/collector.py` | Metrics aggregation, statistics |
| `src/metrics/gpu_monitor.py` | NVIDIA GPU monitoring (pynvml) |
| `src/output/csv_writer.py` | CSV output (summary, detailed, model outputs) |
| `src/output/markdown_writer.py` | Markdown report with embedded images |

### Docker Files

| File | Purpose |
|------|---------|
| `docker/vllm.yaml` | vLLM server compose config |
| `docker/tgi.yaml` | TGI server compose config |
| `docker/tensorrt.yaml` | TensorRT server compose config |
| `docker/prometheus.yaml` | Prometheus monitoring config |

### Scripts

| File | Purpose |
|------|---------|
| `run_all.sh` | Full benchmark runner (start server, benchmark, stop server) |
| `scripts/serve_vllm.sh` | Standalone vLLM server launcher |
| `scripts/serve_tgi.sh` | Standalone TGI server launcher |
| `scripts/serve_trtllm.sh` | Standalone TensorRT server launcher |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | User-facing README |
| `docs/SETUP.md` | Setup and usage guide |
| `docs/benchmark_pipeline.md` | Technical pipeline documentation |

---

## Quick Debugging Commands

```bash
# List available models and platforms
uv run python -m src.cli list

# Show GPU info
uv run python -m src.cli gpu-info

# Validate images folder
uv run python -m src.cli validate --images ./images

# Run single benchmark with debug logging
uv run python -m src.cli benchmark \
  --platform vllm \
  --model qwen2.5-vl-7b \
  --server-url http://localhost:8000 \
  --images ./images \
  --log-level DEBUG
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HuggingFace authentication | (none) |
| `HF_CACHE` | Model cache directory | `~/.cache/huggingface` |
| `MODEL_ID` | Model to serve | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `MAX_MODEL_LEN` | Max context length (vLLM) | `8192` |
| `MAX_INPUT_LEN` | Max input tokens (TGI) | `8192` |
| `GPU_MEM_UTIL` | GPU memory utilization (vLLM) | `0.6` |
| `VLLM_PORT` | vLLM server port | `8000` |
| `TGI_PORT` | TGI server port | `8080` |
