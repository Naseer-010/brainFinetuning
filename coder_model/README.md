# Coder Model — Qwen2.5-Coder-7B-Instruct Fine-tuning

Manim code generation model for JEE Physics/Math animations. Takes structured visual reasoning JSON and produces fully runnable Manim Community Edition Python code.

## Quick Start

```bash
# 1. Setup environment (once)
bash scripts/setup_env.sh

# 2. Activate venv
source ../.venv/bin/activate

# 3. Dry run (check config without training)
bash scripts/train.sh --dry-run

# 4. Train (auto-detects DGX vs local GPU)
bash scripts/train.sh

# 5. Force a specific GPU profile
bash scripts/train.sh --dgx
bash scripts/train.sh --local

# 6. After training — merge & push to HuggingFace
python scripts/merge_and_push.py
python scripts/merge_and_push.py --local-only  # merge without pushing
```

## Configs

| Config | GPU | LoRA | Quant | Batch | Context | Epochs | VRAM |
|--------|-----|------|-------|-------|---------|--------|------|
| `dgx_train.yaml` | A100 80GB | r=64, α=128 | bf16 | 4×8 | 8192 | 5 | ~60GB |
| `local_train.yaml` | RTX 4080+ | r=32, α=64 | 4-bit QLoRA | 1×16 | 4096 | 5 | ~12GB |

> **Note:** The coder model uses 5 epochs (vs 3 for brain) and lower LR (1e-4 vs 2e-4) because Manim API syntax requires deeper memorization and more precise code generation.

## Directory Layout

```
coder_model/
├── configs/
│   ├── dgx_train.yaml      # DGX A100 config
│   └── local_train.yaml     # Local GPU config
├── scripts/
│   ├── setup_env.sh         # Install dependencies
│   ├── train.sh             # Unified trainer
│   └── merge_and_push.py    # LoRA merge + HF push
├── checkpoints/             # (created during training)
│   ├── dgx/
│   └── local/
├── merged/                  # (created after merge)
├── dataset_info.json        # LLaMA-Factory dataset registry
└── README.md
```
