#!/usr/bin/env python3
"""
Manim Code Validator
--------------------
Quality gate: renders each Manim code sample from coder_train.json
in a Docker sandbox (or locally) and filters out any that fail.

Only successfully rendering examples make it into the final training set.

Usage:
    python validate_manim.py \
        --input ../data/training/coder_train.json \
        --output ../data/training/coder_train_validated.json \
        --mode docker                # or "local"
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def render_manim_docker(code: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Render Manim code inside a Docker container.
    Returns (success: bool, error_message: str).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmp_path}:/scene.py:ro",
                "manimcommunity/manim:stable",
                "manim",
                "render",
                "/scene.py",
                "-ql",  # low quality for speed
                "--disable_caching",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        error = result.stderr if not success else ""
        return success, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Render exceeded time limit"
    except FileNotFoundError:
        return False, "Docker not found. Install Docker or use --mode local"
    finally:
        os.unlink(tmp_path)


def render_manim_local(code: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Render Manim code locally (requires manim installed).
    Returns (success: bool, error_message: str).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["manim", "render", tmp_path, "-ql", "--disable_caching"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
        success = result.returncode == 0
        error = result.stderr if not success else ""
        return success, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Render exceeded time limit"
    except FileNotFoundError:
        return False, "Manim not found. Install with: pip install manim"
    finally:
        os.unlink(tmp_path)


def syntax_check_only(code: str) -> tuple[bool, str]:
    """Fast syntax-only check without rendering (good for quick filtering)."""
    try:
        compile(code, "<manim_scene>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def validate_dataset(
    input_path: str,
    output_path: str,
    mode: str = "docker",
    syntax_only: bool = False,
):
    """Validate all entries in a coder training dataset."""
    with open(input_path) as f:
        data = json.load(f)

    print(f"Validating {len(data)} Manim code entries (mode: {mode})\n")

    validated = []
    failed = []

    render_fn = {
        "docker": render_manim_docker,
        "local": render_manim_local,
    }.get(mode)

    for i, entry in enumerate(data):
        code = entry.get("output", "")
        q_num = entry.get("metadata", {}).get("question_number", "?")
        institute = entry.get("metadata", {}).get("institute", "?")
        label = f"[{i + 1}/{len(data)}] Q#{q_num} ({institute})"

        # Always do syntax check first
        syntax_ok, syntax_err = syntax_check_only(code)
        if not syntax_ok:
            print(f"  ✗ {label} — {syntax_err}")
            failed.append({**entry, "_error": syntax_err})
            continue

        if syntax_only:
            print(f"  ✓ {label} — syntax OK")
            validated.append(entry)
            continue

        # Full render check
        success, error = render_fn(code)
        if success:
            print(f"  ✓ {label} — renders OK")
            validated.append(entry)
        else:
            short_err = error.split("\n")[-3:] if error else ["Unknown error"]
            print(f"  ✗ {label} — {'  '.join(short_err)}")
            failed.append({**entry, "_error": error})

    # Save validated
    with open(output_path, "w") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    # Save failures for review
    failed_path = output_path.replace(".json", "_failed.json")
    if failed:
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Passed: {len(validated)} / {len(data)}")
    print(f"  Failed: {len(failed)} / {len(data)}")
    print(
        f"  Pass rate: {len(validated) / len(data) * 100:.1f}%" if data else "  No data"
    )
    print(f"  Output: {output_path}")
    if failed:
        print(f"  Failures: {failed_path}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Validate Manim training data")
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).parent.parent / "data" / "training" / "coder_train.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent.parent
            / "data"
            / "training"
            / "coder_train_validated.json"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["docker", "local"],
        default="docker",
        help="Render mode: docker (sandboxed) or local",
    )
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="Only check Python syntax, skip rendering (fast)",
    )

    args = parser.parse_args()
    validate_dataset(args.input, args.output, args.mode, args.syntax_only)


if __name__ == "__main__":
    main()
