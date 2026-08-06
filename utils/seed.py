"""
utils/seed.py
────────────────────────────────────────────────────────
Deterministic seed control for full reproducibility.

Sets ALL random sources: Python, NumPy, PyTorch (CPU + CUDA),
cuDNN, and DataLoader workers. Call `set_seed()` once at the
start of any script that needs reproducible results.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.backends.cudnn
from typing import Optional


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set ALL random seeds for fully deterministic execution.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, enable cuDNN determinism (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safety

    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        # Non-deterministic but faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # DataLoader worker seed (set via worker_init_fn in get_dataloaders)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader reproducibility.
    Use as: DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Return a deterministically seeded torch.Generator."""
    g = torch.Generator()
    g.manual_seed(seed if seed is not None else 42)
    return g
