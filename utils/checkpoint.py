"""
utils/checkpoint.py – Load trained weights without retraining.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, str]:
    """
    Load model_state_dict from a training or FP16 artifact checkpoint.

    Returns:
        (epoch_or_-1, precision_note)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        epoch = int(state.get("epoch", -1))
    else:
        model.load_state_dict(state)
        epoch = -1

    # Always run scoring math in FP32; GPU forward uses autocast in ProductionDetector.
    model.float()
    precision = "fp16" if "fp16" in checkpoint_path.replace("\\", "/") else "fp32"
    return epoch, precision
