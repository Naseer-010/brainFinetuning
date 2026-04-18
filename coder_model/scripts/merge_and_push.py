#!/usr/bin/env python3
"""
Coder Model — Merge LoRA Adapters & Push to Hugging Face
---------------------------------------------------------
Fuses the trained LoRA weights into the base Qwen2.5-Coder-7B-Instruct model
and uploads the merged model to your Hugging Face repository.

Usage:
    python merge_and_push.py                          # auto-detect checkpoint
    python merge_and_push.py --checkpoint-dir ../checkpoints/dgx
    python merge_and_push.py --local-only             # merge but don't push
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """Find the latest checkpoint directory."""
    cp_path = Path(checkpoints_dir)
    if not cp_path.exists():
        print(f"ERROR: Checkpoint directory not found: {cp_path}", file=sys.stderr)
        sys.exit(1)

    checkpoints = sorted(
        cp_path.glob("checkpoint-*"),
        key=lambda p: (
            int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        ),
        reverse=True,
    )

    if checkpoints:
        return str(checkpoints[0])

    if (cp_path / "adapter_config.json").exists():
        return str(cp_path)

    print(f"ERROR: No checkpoints found in {cp_path}", file=sys.stderr)
    sys.exit(1)


def merge_and_push(
    checkpoint_dir: str,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    hf_repo: str = "",
    output_dir: str = "",
    push: bool = True,
):
    """Merge LoRA adapters into base model and optionally push to HF."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.hf_auth import ensure_hf_auth, get_hf_username

    print(f"═══════════════════════════════════════════════════════════")
    print(f"  💻 Coder Model — Merge & Push")
    print(f"  Base model:   {base_model}")
    print(f"  LoRA adapter: {checkpoint_dir}")
    print(f"═══════════════════════════════════════════════════════════\n")

    # ── Load base model ──
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )

    # ── Load & merge LoRA ──
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # ── Load tokenizer ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # ── Save locally ──
    if not output_dir:
        output_dir = str(Path(checkpoint_dir).parent.parent / "merged")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Push to HF ──
    if push:
        ensure_hf_auth()
        if not hf_repo:
            username = get_hf_username()
            hf_repo = f"{username}/vidgen-coder-7b"

        print(f"\nPushing to Hugging Face: {hf_repo}")
        model.push_to_hub(hf_repo, private=True)
        tokenizer.push_to_hub(hf_repo, private=True)
        print(f"✅ Successfully pushed to: https://huggingface.co/{hf_repo}")
    else:
        print(f"\n✅ Merged model saved locally at: {output_dir}")
        print(
            f"   To push later: python merge_and_push.py --checkpoint-dir {checkpoint_dir}"
        )


def main():
    parser = argparse.ArgumentParser(description="Merge Coder LoRA + Push to HF")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("CODER_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("CODER_HF_REPO", ""),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--local-only", action="store_true")

    args = parser.parse_args()

    if not args.checkpoint_dir:
        model_dir = Path(__file__).resolve().parent.parent
        for profile in ["dgx", "local"]:
            cp_dir = model_dir / "checkpoints" / profile
            if cp_dir.exists():
                args.checkpoint_dir = find_latest_checkpoint(str(cp_dir))
                break
        if not args.checkpoint_dir:
            print(
                "ERROR: No checkpoints found. Train the model first.", file=sys.stderr
            )
            sys.exit(1)

    merge_and_push(
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        hf_repo=args.hf_repo,
        output_dir=args.output_dir,
        push=not args.local_only,
    )


if __name__ == "__main__":
    main()
