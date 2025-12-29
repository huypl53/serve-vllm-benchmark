# VLM Benchmark Setup Guide

Complete guide for setting up and running VLM (Vision Language Model) benchmarks on a remote GPU server.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running Benchmarks](#running-benchmarks)
5. [Configuration](#configuration)
6. [Supported Models](#supported-models)
7. [Platform Details](#platform-details)
8. [Output Format](#output-format)
9. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Clone and setup (one command)
git clone <repo> && cd benchmark-v2 && ./setup.sh

# 2. Copy your images to the server
scp -r /path/to/images user@server:benchmark-v2/images/

# 3. Run benchmark (one command)
./run_all.sh --images ./images --platforms vllm --models qwen2.5-vl-7b

# 4. Download results
scp -r user@server:benchmark-v2/results ./
```

## Requirements

### Hardware

| GPU | VRAM | Recommended Models |
|-----|------|-------------------|
| T4 | 16GB | MiniCPM-V-2.6, Qwen2-VL-7B (with quantization) |
| A10/L4 | 24GB | All 7B models |
| A100/H100 | 40-80GB | All models including Llama-3.2-11B |

### Software

- Python 3.10+
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU drivers (525+)
- HuggingFace account (for gated models like Llama)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
./setup.sh
```

This script will:
- Check Python version
- Verify NVIDIA drivers and GPU
- Create virtual environment
- Install Python dependencies
- Pull Docker images
- Create required directories

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Create directories
mkdir -p images results logs

# Pull Docker images
docker pull vllm/vllm-openai:latest
docker pull ghcr.io/huggingface/text-generation-inference:latest
```

### HuggingFace Token

Some models require authentication:

```bash
# Option 1: Environment variable
export HF_TOKEN=your_token_here

# Option 2: HuggingFace CLI
huggingface-cli login
```

## Running Benchmarks

### Single Benchmark

```bash
source .venv/bin/activate

# Start inference server
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct docker compose -f docker/vllm.yaml up -d

# Wait for server to be ready (check logs)
docker logs -f vlm-benchmark-vllm

# Run benchmark
python -m src.cli benchmark \
  --platform vllm \
  --model qwen2.5-vl-7b \
  --server-url http://localhost:8000 \
  --images ./images \
  --output ./results

# Stop server
docker compose -f docker/vllm.yaml down
```

### Multiple Benchmarks

```bash
# Run with default settings (vLLM + Qwen2.5-VL)
./run_all.sh

# Run all platforms with one model
./run_all.sh --all-platforms --models qwen2.5-vl-7b

# Run all models with one platform
./run_all.sh --platforms vllm --all-models

# Full benchmark (all combinations)
./run_all.sh --all-platforms --all-models --max-images 100
```

### CLI Options

```bash
python -m src.cli benchmark --help

Options:
  --platform        Platform: vllm, tgi, tensorrt
  --model           Model ID from config
  --server-url      Server URL
  --images          Path to images folder
  --output          Output directory
  --prompt          Custom VQA prompt
  --max-images      Limit number of images
  --config          Benchmark config file
  --models-config   Models config file
  --log-level       DEBUG, INFO, WARNING, ERROR
```

### Useful Commands

```bash
# List available models and platforms
python -m src.cli list

# Show GPU information
python -m src.cli gpu-info

# Validate images folder
python -m src.cli validate --images ./images
```

## Configuration

### Benchmark Settings (`config/benchmark.yaml`)

```yaml
benchmark:
  warmup_iterations: 2      # Warmup runs (not counted)
  num_iterations: 3         # Iterations per image
  max_new_tokens: 256       # Max generated tokens
  temperature: 0.7          # Sampling temperature
  max_retries: 3            # Retry on failure

output:
  save_model_outputs: true  # Save generated text for review
```

### Model Configuration (`config/models.yaml`)

Add custom models by editing this file:

```yaml
models:
  my-custom-model:
    huggingface_id: "organization/model-name"
    display_name: "My Model"
    memory_estimate_gb: 16
    min_vram_gb: 16
    platforms:
      vllm:
        supported: true
        args:
          trust_remote_code: true
          max_model_len: 8192
```

## Supported Models

| Model | vLLM | TGI | TensorRT | VRAM |
|-------|------|-----|----------|------|
| Qwen2.5-VL-7B | Yes | Yes | Yes | ~16GB |
| Qwen2-VL-7B | Yes | Yes | Yes (best) | ~15GB |
| Llama-3.2-Vision-11B | Yes | Yes | Yes | ~24GB |
| MiniCPM-V-2.6 | Yes | Yes | No | ~8GB |

## Platform Details

### vLLM (Recommended)

- **Best for**: General use, best VLM support
- **Pros**: Easy setup, OpenAI-compatible API, good performance
- **Image**: `vllm/vllm-openai:latest`

```bash
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct docker compose -f docker/vllm.yaml up -d
```

### TGI (Text Generation Inference)

- **Best for**: HuggingFace ecosystem integration
- **Note**: In maintenance mode as of Dec 2025
- **Image**: `ghcr.io/huggingface/text-generation-inference:latest`

```bash
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct docker compose -f docker/tgi.yaml up -d
```

### TensorRT-LLM

- **Best for**: Maximum performance on NVIDIA GPUs
- **Note**: Limited VLM support (mainly Qwen2-VL)
- **Image**: `nvcr.io/nvidia/tensorrt-llm:latest`

```bash
MODEL_ID=Qwen/Qwen2-VL-7B-Instruct docker compose -f docker/tensorrt.yaml up -d
```

## Output Format

### Summary CSV (`results/benchmark_summary.csv`)

Combined results from all benchmark runs:

| Column | Description |
|--------|-------------|
| platform | Platform name |
| model_id | Model identifier |
| avg_ttft_ms | Average time to first token |
| p95_ttft_ms | 95th percentile TTFT |
| avg_total_latency_ms | Average total latency |
| p95_total_latency_ms | 95th percentile latency |
| avg_tokens_per_second | Average throughput |
| peak_vram_mb | Peak GPU memory usage |

### Detailed Results (`results/detailed_results_*.csv`)

Per-image, per-iteration results for analysis.

### Model Outputs (`results/model_outputs_*.csv`)

Generated text for quality review.

## Troubleshooting

### Server Won't Start

```bash
# Check Docker logs
docker logs vlm-benchmark-vllm

# Check GPU memory
nvidia-smi

# Reduce model size or use quantization
export GPU_MEM_UTIL=0.8
```

### Out of Memory

1. Reduce `max_model_len` in config
2. Use a smaller model
3. Enable quantization (for supported models)
4. Reduce `gpu_memory_utilization`

### Connection Refused

```bash
# Check if server is running
docker ps

# Check server health
curl http://localhost:8000/v1/models  # vLLM
curl http://localhost:8080/health     # TGI
```

### Slow Performance

1. Ensure GPU is being used (check `nvidia-smi` during inference)
2. Increase warmup iterations
3. Check for thermal throttling
4. Use TensorRT for best performance (limited model support)

### HuggingFace Authentication

```bash
# For gated models like Llama
export HF_TOKEN=your_token

# Or login interactively
huggingface-cli login

# Accept model license on HuggingFace website
# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
```

## Advanced Usage

### Custom Prompts

```bash
python -m src.cli benchmark \
  --prompt "What objects are visible in this image? List them." \
  ...
```

### Batch Size Tuning

Edit `config/benchmark.yaml`:

```yaml
benchmark:
  num_iterations: 5  # More iterations for statistical significance
```

### GPU Selection

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=1
docker compose -f docker/vllm.yaml up -d
```

## Performance Tips

1. **Use vLLM for most cases** - best balance of speed and compatibility
2. **Warm up before benchmarking** - first few inferences are slower
3. **Use SSD storage** - model loading is I/O bound
4. **Pre-download models** - avoid download time in benchmarks
5. **Monitor GPU thermals** - throttling affects results

## Contact & Support

For issues, please open a GitHub issue with:
- GPU model and VRAM
- Error logs
- Steps to reproduce
