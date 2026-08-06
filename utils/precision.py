"""Numerically safe mixed-precision policy for CHAAD."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def cuda_bf16_available() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def safe_autocast(device: torch.device | str, enabled: bool = True):
    """Use CUDA BF16 when supported; otherwise prefer stable FP32."""
    resolved = torch.device(device)
    if enabled and resolved.type == "cuda" and cuda_bf16_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
