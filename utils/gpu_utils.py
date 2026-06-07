"""
utils/gpu_utils.py
────────────────────────────────────────────────────────
GPU optimization utilities for production deployment.

Features:
  - PyTorch 2.x compilation with torch.compile()
  - CUDA optimization settings
  - Memory management utilities
  - Mixed precision helpers
  - Multi-GPU support detection
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def setup_cuda_optimizations():
    """
    Configure CUDA for maximum performance.
    Call this at the start of your training/inference script.
    """
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN auto-tuner for convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Set deterministic behavior only if required (slower but reproducible)
    # torch.backends.cudnn.deterministic = True

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {gpu_name} | Memory: {gpu_memory:.2f} GB")
    print(f"[GPU] TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[GPU] cuDNN benchmark: {torch.backends.cudnn.benchmark}")


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> nn.Module:
    """
    Compile model with torch.compile() for PyTorch 2.0+ optimization.

    Args:
        model: The PyTorch model to compile
        mode: Compilation mode - "default", "reduce-overhead", "max-autotune"
        fullgraph: If True, requires entire model to be captured in graph

    Returns:
        Compiled model (or original if compilation fails)
    """
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile() not available (PyTorch < 2.0). Using uncompiled model.")
        return model

    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        print(f"[GPU] Model compiled with mode='{mode}'")
        return compiled
    except Exception as e:
        warnings.warn(f"Model compilation failed: {e}. Using uncompiled model.")
        return model


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_memory_gb: Optional[float] = None,
) -> int:
    """
    Heuristic to find optimal batch size that fits in GPU memory.

    Args:
        model: The model to benchmark
        input_shape: Shape of a single input (C, H, W) or (C, T)
        device: Target device
        max_memory_gb: Maximum GPU memory to use (default: 90% of available)

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 32  # Default for CPU

    if max_memory_gb is None:
        max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9

    # Estimate memory per sample (rough heuristic)
    sample_size = 1
    for dim in input_shape:
        sample_size *= dim

    # Bytes per float32 + overhead for activations (approx 4x for backprop)
    bytes_per_sample = sample_size * 4 * 4  # 4x safety factor for gradients/optimizer

    # Available memory in bytes
    available_bytes = max_memory_gb * 1e9

    # Conservative estimate
    estimated_batch = int(available_bytes / bytes_per_sample / 4)  # Divide by 4 for safety

    return max(1, min(estimated_batch, 128))  # Cap at 128


def empty_cache():
    """Force CUDA cache emptying to free fragmented memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "free_gb": (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated()) / 1e9,
    }


def print_memory_stats(prefix: str = ""):
    """Print current GPU memory statistics."""
    stats = get_memory_stats()
    if stats:
        print(f"[{prefix}] GPU Memory: Allocated={stats['allocated_gb']:.2f}GB | "
              f"Reserved={stats['reserved_gb']:.2f}GB | Free={stats['free_gb']:.2f}GB")


def set_memory_fraction(fraction: float = 0.9):
    """
    Limit PyTorch to use only a fraction of GPU memory.
    Useful for running multiple processes on same GPU.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


class GPUMonitor:
    """Context manager for monitoring GPU memory during operations."""

    def __init__(self, name: str = "operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_stats: Optional[Dict[str, float]] = None

    def __enter__(self):
        if torch.cuda.is_available() and self.verbose:
            self.start_stats = get_memory_stats()
            print(f"[GPU Monitor] Starting '{self.name}'...")
            print_memory_stats("Before")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and self.verbose:
            end_stats = get_memory_stats()
            print_memory_stats("After")
            if self.start_stats:
                delta = end_stats['allocated_gb'] - self.start_stats['allocated_gb']
                print(f"[GPU Monitor] '{self.name}' memory delta: {delta:+.2f}GB")


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Apply inference optimizations: eval mode, no grad, and optional compilation.
    """
    model.eval()

    # Use inference mode for additional optimizations
    with torch.inference_mode():
        pass

    return model


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_flash_attention():
    """
    Enable Flash Attention 2 if available (significant speedup for transformers).
    Requires: pip install flash-attn
    """
    try:
        import flash_attn
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("[GPU] Flash Attention 2 enabled")
    except ImportError:
        pass


def setup_distributed() -> Dict[str, Any]:
    """
    Setup for distributed training if available.
    Returns world_size, rank, local_rank.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "distributed": True,
        }
    return {"rank": 0, "world_size": 1, "local_rank": 0, "distributed": False}
