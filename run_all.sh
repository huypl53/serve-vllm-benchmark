#!/bin/bash
# VLM Benchmark Runner - One-command full benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
IMAGE_DIR="./images"
OUTPUT_DIR="./results"
MAX_IMAGES=""
PROMPT=""
CONCURRENCY=""  # Number of concurrent requests (empty = use config default)
PLATFORMS="vllm"  # Start with vLLM only by default
MODELS="qwen2.5-vl-7b"  # Start with one model
VLLM_PORT=8000
TGI_PORT=8080
TENSORRT_PORT=8001

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --images)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --all-platforms)
            PLATFORMS="vllm,tgi,tensorrt"
            shift
            ;;
        --all-models)
            MODELS="qwen2.5-vl-7b,qwen2.5-vl-7b-w8a8,qwen2.5-vl-3b,qwen2.5-vl-3b-w4a16,qwen3-vl-8b,qwen3-vl-4b,internvl3.5-2b,internvl3.5-4b,internvl3.5-1b,erax-vl-2b,vintern-1b,vistral-7b-chat,pangea-7b,lavy-instruct"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --images DIR        Image folder (default: ./images)"
            echo "  --output DIR        Output folder (default: ./results)"
            echo "  --max-images N      Max images to process"
            echo "  --prompt TEXT       VQA prompt"
            echo "  --concurrency N     Number of concurrent requests (default: 1)"
            echo "  --platforms LIST    Comma-separated platforms (default: vllm)"
            echo "  --models LIST       Comma-separated models (default: qwen2.5-vl-7b)"
            echo "  --all-platforms     Test all platforms (vllm,tgi,tensorrt)"
            echo "  --all-models        Test all models"
            echo ""
            echo "Example:"
            echo "  $0 --all-platforms --all-models --max-images 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              VLM Benchmark Runner                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo -e "${RED}Virtual environment not found. Run ./setup.sh first.${NC}"
    exit 1
fi

# Preflight checks
echo -e "${BLUE}==> Running preflight checks...${NC}"

# Check images folder
if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${RED}✗ Image folder not found: $IMAGE_DIR${NC}"
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No images found in $IMAGE_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $IMAGE_COUNT images in $IMAGE_DIR${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU not available${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1 | tr -d ' MiB')
echo -e "${GREEN}✓ GPU: $GPU_NAME (Free: ${GPU_MEMORY}MB)${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to start server
start_server() {
    local platform=$1
    local model=$2

    echo -e "${BLUE}==> Starting $platform server for $model...${NC}"

    # Get HuggingFace model ID from models.yaml (single source of truth)
    local hf_id
    hf_id=$(python -c "
import yaml
with open('$SCRIPT_DIR/config/models.yaml') as f:
    data = yaml.safe_load(f)
model_config = data.get('models', {}).get('$model', {})
print(model_config.get('huggingface_id', '$model'))
")
    if [ -z "$hf_id" ]; then
        hf_id=$model
    fi

    export MODEL_ID="$hf_id"

    case $platform in
        vllm)
            docker compose -f docker/vllm.yaml up -d
            wait_for_server "http://localhost:$VLLM_PORT/v1/models"
            ;;
        tgi)
            docker compose -f docker/tgi.yaml up -d
            wait_for_server "http://localhost:$TGI_PORT/health"
            ;;
        tensorrt)
            docker compose -f docker/tensorrt.yaml --profile serve up -d
            wait_for_server "http://localhost:$TENSORRT_PORT/v1/models"
            ;;
    esac
}

# Function to wait for server
wait_for_server() {
    local url=$1
    local max_wait=900
    local wait_time=0

    echo -n "Waiting for server to be ready..."
    while [ $wait_time -lt $max_wait ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 5
        wait_time=$((wait_time + 5))
    done

    echo -e " ${RED}Timeout!${NC}"
    return 1
}

# Function to stop server
stop_server() {
    local platform=$1
    echo -e "${BLUE}==> Stopping $platform server...${NC}"

    case $platform in
        vllm) docker compose -f docker/vllm.yaml down ;;
        tgi) docker compose -f docker/tgi.yaml down ;;
        tensorrt) docker compose -f docker/tensorrt.yaml --profile serve down ;;
    esac
}

# Function to run benchmark
run_benchmark() {
    local platform=$1
    local model=$2
    local server_url=$3

    echo -e "${CYAN}==> Benchmarking $model on $platform${NC}"

    local cmd="python -m src.cli benchmark"
    cmd="$cmd --platform $platform"
    cmd="$cmd --model $model"
    cmd="$cmd --server-url $server_url"
    cmd="$cmd --images $IMAGE_DIR"
    cmd="$cmd --output $OUTPUT_DIR"

    if [ -n "$MAX_IMAGES" ]; then
        cmd="$cmd --max-images $MAX_IMAGES"
    fi

    if [ -n "$PROMPT" ]; then
        cmd="$cmd --prompt \"$PROMPT\""
    fi

    if [ -n "$CONCURRENCY" ]; then
        cmd="$cmd --concurrency $CONCURRENCY"
    fi

    eval $cmd
}

# Main benchmark loop
IFS=',' read -ra PLATFORM_LIST <<< "$PLATFORMS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"

TOTAL_BENCHMARKS=$((${#PLATFORM_LIST[@]} * ${#MODEL_LIST[@]}))
CURRENT=0
SUCCESSFUL=0
FAILED=0

echo ""
echo -e "${BLUE}Running $TOTAL_BENCHMARKS benchmark(s)...${NC}"
echo ""

for platform in "${PLATFORM_LIST[@]}"; do
    # Get server URL
    case $platform in
        vllm) SERVER_URL="http://localhost:$VLLM_PORT" ;;
        tgi) SERVER_URL="http://localhost:$TGI_PORT" ;;
        tensorrt) SERVER_URL="http://localhost:$TENSORRT_PORT" ;;
    esac

    for model in "${MODEL_LIST[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "Benchmark $CURRENT/$TOTAL_BENCHMARKS: $model on $platform"
        echo "════════════════════════════════════════════════════════════"

        # Check if model is supported on platform
        # (This is a simplified check - the CLI will do full validation)
        if [ "$platform" = "tensorrt" ] && [ "$model" = "minicpm-v-2.6" ]; then
            echo -e "${YELLOW}Skipping: $model not supported on tensorrt${NC}"
            continue
        fi

        # Start server if not running
        if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1 && \
           ! curl -s "$SERVER_URL/v1/models" > /dev/null 2>&1 && \
           ! curl -s "$SERVER_URL/v2/health/ready" > /dev/null 2>&1; then
            start_server "$platform" "$model"
        fi

        # Run benchmark
        if run_benchmark "$platform" "$model" "$SERVER_URL"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
            echo -e "${GREEN}✓ Completed: $model on $platform${NC}"
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}✗ Failed: $model on $platform${NC}"
        fi

        # Stop server after each model to free memory
        stop_server "$platform"
    done
done

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  Benchmark Complete!                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "Total: $TOTAL_BENCHMARKS | ${GREEN}Successful: $SUCCESSFUL${NC} | ${RED}Failed: $FAILED${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View combined summary:"
echo "  cat $OUTPUT_DIR/benchmark_summary.csv"
echo ""
