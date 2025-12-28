#!/bin/bash
# VLM Benchmark Setup Script
# One-command setup for remote GPU server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Header
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           VLM Benchmark Setup Script                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
print_step "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python3 not found"
    exit 1
fi

# Check Docker
print_step "Checking Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    print_success "Docker $DOCKER_VERSION found"
else
    print_warning "Docker not found. Install Docker for containerized inference servers."
fi

# Check NVIDIA drivers
print_step "Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    print_success "NVIDIA GPU detected: $GPU_NAME"
    print_success "GPU Memory: $GPU_MEMORY"
    print_success "Driver Version: $DRIVER_VERSION"

    # Detect GPU type and give recommendations
    if [[ "$GPU_NAME" == *"T4"* ]]; then
        print_warning "T4 detected (16GB). Large models may need quantization."
    elif [[ "$GPU_NAME" == *"A10"* ]] || [[ "$GPU_NAME" == *"L4"* ]]; then
        print_success "A10/L4 detected (24GB). Should run most 7B models."
    elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
        print_success "High-end GPU detected. Can run all models."
    fi
else
    print_error "NVIDIA drivers not found. GPU acceleration won't work."
    exit 1
fi

# Check NVIDIA Container Toolkit
print_step "Checking NVIDIA Container Toolkit..."
if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    print_success "NVIDIA Container Toolkit is configured"
else
    print_warning "NVIDIA Container Toolkit may not be configured. Docker GPU support might not work."
    print_warning "Install with: distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list && sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

# Create virtual environment
print_step "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate and install dependencies
print_step "Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q
print_success "Dependencies installed"

# Create directories
print_step "Creating directories..."
mkdir -p images results logs
print_success "Created directories: images/, results/, logs/"

# Check HuggingFace token
print_step "Checking HuggingFace token..."
if [ -n "$HUGGING_FACE_HUB_TOKEN" ] || [ -n "$HF_TOKEN" ]; then
    print_success "HuggingFace token found in environment"
elif [ -f ~/.huggingface/token ] || [ -f ~/.cache/huggingface/token ]; then
    print_success "HuggingFace token found in cache"
else
    print_warning "No HuggingFace token found. Some models may require authentication."
    echo "  Set with: export HF_TOKEN=your_token_here"
    echo "  Or run: huggingface-cli login"
fi

# Pull Docker images (optional)
if command -v docker &> /dev/null; then
    print_step "Pulling Docker images (this may take a while)..."

    echo "  Pulling vLLM image..."
    docker pull vllm/vllm-openai:latest 2>/dev/null && print_success "vLLM image ready" || print_warning "Failed to pull vLLM image"

    echo "  Pulling TGI image..."
    docker pull ghcr.io/huggingface/text-generation-inference:latest 2>/dev/null && print_success "TGI image ready" || print_warning "Failed to pull TGI image"

    # TensorRT-LLM image is large, make it optional
    read -p "Pull TensorRT-LLM image (large, ~20GB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker pull nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 2>/dev/null && print_success "TensorRT image ready" || print_warning "Failed to pull TensorRT image"
    else
        print_warning "Skipped TensorRT image"
    fi
fi

# Verify installation
print_step "Verifying installation..."
source .venv/bin/activate
python3 -c "from src.cli import main; print('CLI import OK')" && print_success "CLI module OK"
python3 -c "from src.platforms import VLLMPlatform, TGIPlatform, TensorRTPlatform; print('Platforms import OK')" && print_success "Platform modules OK"
python3 -c "from src.metrics.gpu_monitor import get_gpu_info; info = get_gpu_info(); print(f'GPU monitoring OK: {info[\"gpus\"][0][\"name\"] if info else \"N/A\"}')" && print_success "GPU monitoring OK"

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your images to the 'images/' folder:"
echo "   scp -r /path/to/your/images/* user@server:$(pwd)/images/"
echo ""
echo "2. Start an inference server (choose one):"
echo ""
echo "   # vLLM (recommended):"
echo "   docker compose -f docker/vllm.yaml up -d"
echo ""
echo "   # TGI:"
echo "   docker compose -f docker/tgi.yaml up -d"
echo ""
echo "3. Run the benchmark:"
echo "   source .venv/bin/activate"
echo "   python -m src.cli benchmark \\"
echo "     --platform vllm \\"
echo "     --model qwen2.5-vl-7b \\"
echo "     --server-url http://localhost:8000 \\"
echo "     --images ./images"
echo ""
echo "   Or run all benchmarks:"
echo "   ./run_all.sh"
echo ""
echo "4. Results will be saved to 'results/' folder"
echo ""
