#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Brain Model - Environment Setup                            ║
# ║  Installs LLaMA-Factory + dependencies                     ║
# ╚══════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODEL_DIR")"
VENV_DIR="$ROOT_DIR/.venv"

echo "═══════════════════════════════════════════════════════════"
echo "  🧠 Brain Model — Environment Setup"
echo "═══════════════════════════════════════════════════════════"

# ─── Create venv if it doesn't exist ─────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# ─── Activate venv ──────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python3)"
echo "Python version: $(python3 --version)"

# ─── Upgrade pip ────────────────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel

# ─── Install PyTorch (detect CUDA version) ──────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "NVIDIA driver detected: $CUDA_VER"
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "No NVIDIA GPU found, installing CPU PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ─── Install LLaMA-Factory ─────────────────────────────────────────────────
echo ""
echo "Installing LLaMA-Factory..."
if [ -d "$ROOT_DIR/LLaMA-Factory" ]; then
    echo "LLaMA-Factory repo already exists, pulling latest..."
    cd "$ROOT_DIR/LLaMA-Factory" && git pull
else
    echo "Cloning LLaMA-Factory..."
    cd "$ROOT_DIR"
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

cd "$ROOT_DIR/LLaMA-Factory"
pip install -e ".[torch,metrics,bitsandbytes,qwen]"

# ─── Install additional dependencies ────────────────────────────────────────
echo ""
echo "Installing additional dependencies..."
pip install \
    peft>=0.11.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    huggingface_hub>=0.23.0 \
    python-dotenv>=1.0.0 \
    tensorboard>=2.14.0 \
    flash-attn --no-build-isolation 2>/dev/null || echo "  ⚠ flash-attn install failed (may need manual build)"

# ─── Verify installation ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Verifying installation..."
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:          {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

python3 -c "import peft; print(f'  PEFT:          {peft.__version__}')"
python3 -c "import transformers; print(f'  Transformers:  {transformers.__version__}')"

# Check LLaMA-Factory
if command -v llamafactory-cli &> /dev/null; then
    echo "  LLaMA-Factory: ✓ installed"
else
    echo "  LLaMA-Factory: ✗ not found in PATH"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "  Train with:    bash $SCRIPT_DIR/train.sh"
echo "═══════════════════════════════════════════════════════════"
