# Brain Model — Qwen2.5-VL-7B-Instruct Fine-tuning

Visual reasoning model for JEE Physics/Math. Takes questions with diagrams and produces structured JSON schemas with `visual_type`, `narration`, `step_by_step` reasoning, and `final_answer`.

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

| Config | GPU | LoRA | Quant | Batch | VRAM |
|--------|-----|------|-------|-------|------|
| `dgx_train.yaml` | A100 80GB | r=64, α=128 | bf16 | 4×8 | ~60GB |
| `local_train.yaml` | RTX 4080+ | r=32, α=64 | 4-bit QLoRA | 1×16 | ~12GB |

## Directory Layout

```
brain_model/
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
