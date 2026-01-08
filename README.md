# VLM Benchmark

Benchmark Vision Language Models (VLMs) across multiple inference platforms with minimal setup.

## Features

- **Multi-platform support**: vLLM, TGI (Text Generation Inference), TensorRT-LLM
- **Multiple VLM models**: Qwen2.5-VL, Qwen3-VL, InternVL3.5, EraX-VL, Vintern, Vistral-7B-Chat, Pangea-7B, Lavy-instruct
- **One-command setup**: `./setup.sh` handles everything
- **Comprehensive metrics**: TTFT, latency, tokens/sec, VRAM usage
- **Rich output**: CSV results, markdown reports with embedded images, model outputs for review
- **GPU auto-detection**: Recommends models based on available VRAM
- **Cache bypass**: vLLM image cache bypass for accurate multi-iteration benchmarks

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
| Qwen3-VL-8B | ~20GB | vLLM |
| InternVL3.5-4B | ~10GB | vLLM |
| Vistral-7B-Chat | ~16GB | vLLM |
| Pangea-7B | ~16GB | vLLM |
| Lavy-instruct | ~8GB | vLLM |
| EraX-VL-2B | ~6GB | vLLM |
| Vintern-1B | ~4GB | vLLM |

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
- `example_responses_*.md` - Rich markdown with embedded images and performance metrics

## Documentation

- **[docs/benchmark_pipeline.md](docs/benchmark_pipeline.md)** - Complete benchmark pipeline guide, backend recommendations, and performance tuning
- **[docs/SETUP.md](docs/SETUP.md)** - Setup guide and troubleshooting

## License

MIT
