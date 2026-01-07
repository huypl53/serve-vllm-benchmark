# VLM Benchmark

Benchmark Vision Language Models (VLMs) across multiple inference platforms with minimal setup.

## Features

- **Multi-platform support**: vLLM, TGI (Text Generation Inference), TensorRT-LLM
- **Multiple VLM models**: Qwen2.5-VL, Qwen2-VL, Llama-3.2-Vision, MiniCPM-V
- **One-command setup**: `./setup.sh` handles everything
- **Comprehensive metrics**: TTFT, latency, tokens/sec, VRAM usage
- **CSV output**: Detailed results, summaries, and model outputs for review
- **GPU auto-detection**: Recommends models based on available VRAM

## Quick Start

```bash
# 1. Setup (one command)
./setup.sh

# 2. Add your images
cp /path/to/images/* ./images/

# 3. Run benchmark
./run_all.sh --images ./images

# 4. View results
cat results/benchmark_summary.csv
```

## Requirements

- Python 3.10+
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (T4/A10/L4/A100)

## Supported Models

| Model | VRAM | Best Platform |
|-------|------|---------------|
| Qwen2.5-VL-7B | ~16GB | vLLM |
| Qwen2-VL-7B | ~15GB | vLLM/TensorRT |
| Llama-3.2-Vision-11B | ~24GB | vLLM |
| MiniCPM-V-2.6 | ~8GB | vLLM |

## Usage

### Single Benchmark

```bash
source .venv/bin/activate

# Start server
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct docker compose -f docker/vllm.yaml up -d

# Run benchmark
python -m src.cli benchmark \
  --platform vllm \
  --model qwen2.5-vl-7b \
  --server-url http://localhost:8000 \
  --images ./images

# Stop server
docker compose -f docker/vllm.yaml down
```

### All Benchmarks

```bash
# Test all platforms with all models
./run_all.sh --all-platforms --all-models --max-images 100
```

## Output

Results are saved to `results/`:

- `benchmark_summary.csv` - Combined metrics from all runs
- `detailed_results_*.csv` - Per-image, per-iteration results
- `model_outputs_*.csv` - Generated text for quality review

## Documentation

See [docs/SETUP.md](docs/SETUP.md) for complete setup guide and troubleshooting.

## License

MIT
