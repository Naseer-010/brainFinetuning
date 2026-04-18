#!/usr/bin/env python3
"""
GPU Detection Utility
---------------------
Auto-detects whether running on a DGX A100 cluster or a local consumer GPU.
Returns the appropriate training profile so scripts can auto-select configs.

Usage:
    from shared.gpu_detect import detect_gpu_profile, get_gpu_info

    profile = detect_gpu_profile()  # "dgx" or "local"
    info = get_gpu_info()           # dict with gpu details
"""

import os
import subprocess
import json
import sys
from typing import Optional


# GPU names that indicate a DGX / datacenter environment
DGX_GPU_PATTERNS = [
    "A100",
    "H100",
    "H200",
    "A800",
    "H800",
    "B200",
]

# Minimum VRAM (in MiB) to consider a GPU "datacenter-class"
DGX_VRAM_THRESHOLD_MIB = 40_000  # 40 GB


def _run_nvidia_smi(query: str) -> Optional[str]:
    """Run nvidia-smi with a specific query and return stdout."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_gpu_info() -> dict:
    """
    Return a dict with GPU information:
    {
        "available": bool,
        "count": int,
        "gpus": [{"name": str, "vram_mib": int, "index": int}, ...],
        "cuda_version": str | None,
        "driver_version": str | None,
    }
    """
    info = {
        "available": False,
        "count": 0,
        "gpus": [],
        "cuda_version": None,
        "driver_version": None,
    }

    # Check GPU names and memory
    names_raw = _run_nvidia_smi("name")
    mem_raw = _run_nvidia_smi("memory.total")

    if names_raw is None:
        return info

    names = [n.strip() for n in names_raw.split("\n") if n.strip()]
    mems = [m.strip() for m in (mem_raw or "").split("\n") if m.strip()]

    info["available"] = len(names) > 0
    info["count"] = len(names)

    for i, name in enumerate(names):
        vram = int(mems[i]) if i < len(mems) and mems[i].isdigit() else 0
        info["gpus"].append({"name": name, "vram_mib": vram, "index": i})

    # CUDA and driver version
    driver_raw = _run_nvidia_smi("driver_version")
    if driver_raw:
        info["driver_version"] = driver_raw.split("\n")[0].strip()

    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.split("\n"):
                if "release" in line.lower():
                    # e.g., "Cuda compilation tools, release 12.2, V12.2.140"
                    parts = line.split("release")[-1].strip().split(",")
                    info["cuda_version"] = parts[0].strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def detect_gpu_profile() -> str:
    """
    Detect GPU profile. Returns "dgx" or "local".

    Priority:
      1. GPU_PROFILE env var (explicit override)
      2. Auto-detect from nvidia-smi output
      3. Fallback to "local"
    """
    # 1. Check for manual override
    override = os.environ.get("GPU_PROFILE", "").strip().lower()
    if override in ("dgx", "local"):
        return override

    # 2. Auto-detect
    info = get_gpu_info()
    if not info["available"]:
        print(
            "[gpu_detect] WARNING: No NVIDIA GPU found, defaulting to 'local' profile",
            file=sys.stderr,
        )
        return "local"

    # Check if any GPU matches DGX patterns
    for gpu in info["gpus"]:
        gpu_name = gpu["name"].upper()
        for pattern in DGX_GPU_PATTERNS:
            if pattern in gpu_name:
                return "dgx"

        # Also check by VRAM — any GPU with 40GB+ is likely datacenter
        if gpu["vram_mib"] >= DGX_VRAM_THRESHOLD_MIB:
            return "dgx"

    return "local"


def print_gpu_summary():
    """Print a human-readable summary of GPU detection results."""
    info = get_gpu_info()
    profile = detect_gpu_profile()

    print("=" * 60)
    print("  GPU Detection Summary")
    print("=" * 60)
    print(f"  Profile:        {profile.upper()}")
    print(f"  GPUs found:     {info['count']}")
    print(f"  Driver:         {info['driver_version'] or 'N/A'}")
    print(f"  CUDA:           {info['cuda_version'] or 'N/A'}")

    for gpu in info["gpus"]:
        print(f"  [{gpu['index']}] {gpu['name']}  |  {gpu['vram_mib']} MiB VRAM")

    print("=" * 60)


if __name__ == "__main__":
    print_gpu_summary()
    profile = detect_gpu_profile()
    print(f"\nDetected profile: {profile}")
    # Also output as JSON for scripting
    print(json.dumps({"profile": profile, **get_gpu_info()}, indent=2))
