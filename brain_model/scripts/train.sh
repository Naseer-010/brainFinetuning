#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Brain Model - Unified Training Launcher                    ║
# ║  Auto-detects DGX A100 vs local GPU and picks the config    ║
# ╚══════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODEL_DIR")"

# ─── Parse arguments ────────────────────────────────────────────────────────
DRY_RUN=false
FORCE_PROFILE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --dgx)      FORCE_PROFILE="dgx" ;;
        --local)    FORCE_PROFILE="local" ;;
        --help|-h)
            echo "Usage: bash train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run     Print config and exit without training"
            echo "  --dgx         Force DGX config"
            echo "  --local       Force local GPU config"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
    esac
done

# ─── Detect GPU profile ─────────────────────────────────────────────────────
if [ -n "$FORCE_PROFILE" ]; then
    PROFILE="$FORCE_PROFILE"
else
    PROFILE=$(python3 -c "
import sys
sys.path.insert(0, '$ROOT_DIR')
from shared.gpu_detect import detect_gpu_profile
print(detect_gpu_profile())
")
fi

echo "═══════════════════════════════════════════════════════════"
echo "  🧠 Brain Model Training"
echo "  GPU Profile:  $PROFILE"
echo "═══════════════════════════════════════════════════════════"

# ─── Select config ───────────────────────────────────────────────────────────
if [ "$PROFILE" = "dgx" ]; then
    CONFIG="$MODEL_DIR/configs/dgx_train.yaml"
    echo "  Config:       DGX A100 (LoRA r=64, bf16, batch=4×8)"
else
    CONFIG="$MODEL_DIR/configs/local_train.yaml"
    echo "  Config:       Local GPU (QLoRA r=32, 4-bit, batch=1×16)"
fi

echo "  Config file:  $CONFIG"
echo "═══════════════════════════════════════════════════════════"

# ─── GPU summary ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

# ─── Dry run: show config and exit ──────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo "─── DRY RUN: Config contents ───"
    cat "$CONFIG"
    echo ""
    echo "─── Command that would run ───"
    echo "llamafactory-cli train $CONFIG"
    echo ""
    echo "Exiting (dry run, no training started)."
    exit 0
fi

# ─── Copy dataset_info.json to the dataset dir ─────────────────────────────
DATASET_DIR="$ROOT_DIR/data/training"
mkdir -p "$DATASET_DIR"

if [ -f "$MODEL_DIR/dataset_info.json" ]; then
    cp "$MODEL_DIR/dataset_info.json" "$DATASET_DIR/dataset_info.json"
    echo "  Copied dataset_info.json → $DATASET_DIR/"
fi

# ─── Launch training ────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting training..."
echo ""

llamafactory-cli train "$CONFIG"

echo ""
echo "✅ Brain model training complete!"
echo "   Checkpoints: $MODEL_DIR/checkpoints/$PROFILE/"
