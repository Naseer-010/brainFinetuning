#!/usr/bin/env python3
"""
Hugging Face Authentication Helper
-----------------------------------
Reads HF_TOKEN from environment or .env file and authenticates with the Hub.

Usage:
    from shared.hf_auth import ensure_hf_auth

    ensure_hf_auth()  # Authenticate before push operations
"""

import os
import sys
import subprocess
from pathlib import Path


def _load_dotenv():
    """Load .env file from project root if python-dotenv is available."""
    # Walk up from this file to find .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # Manual fallback if python-dotenv isn't installed
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value


def get_hf_token() -> str:
    """Get the HuggingFace token from environment."""
    _load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(
            "[hf_auth] ERROR: HF_TOKEN not found.\n"
            "  Set it in your .env file or export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def ensure_hf_auth():
    """
    Authenticate with Hugging Face Hub.
    Uses the huggingface_hub Python library (preferred) or CLI fallback.
    """
    token = get_hf_token()

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=True)
        print("[hf_auth] Successfully authenticated with Hugging Face Hub")
    except ImportError:
        # Fallback to CLI
        result = subprocess.run(
            ["huggingface-cli", "login", "--token", token, "--add-to-git-credential"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[hf_auth] Successfully authenticated via CLI")
        else:
            print(f"[hf_auth] ERROR: CLI auth failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)


def get_hf_username() -> str:
    """Get the HuggingFace username from environment."""
    _load_dotenv()
    username = os.environ.get("HF_USERNAME", "").strip()
    if not username:
        print(
            "[hf_auth] ERROR: HF_USERNAME not found.\n"
            "  Set it in your .env file or export HF_USERNAME=your_username",
            file=sys.stderr,
        )
        sys.exit(1)
    return username


if __name__ == "__main__":
    ensure_hf_auth()
    print(f"Authenticated as: {get_hf_username()}")
