"""
paths.py – Canonical artifact locations (inference-only).

Do not retrain or recalibrate: use the checkpoints and artifacts already on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved paths for production inference."""

    root: str = _ROOT
    checkpoint_dir: str = os.path.join(_ROOT, "checkpoints")
    artifacts_dir: str = os.path.join(_ROOT, "artifacts", "models")
    edge_models_dir: str = os.path.join(_ROOT, "edge_deploy", "models")

    best_model_fp32: str = os.path.join(_ROOT, "checkpoints", "best_model.pt")
    best_model_fp16: str = os.path.join(_ROOT, "artifacts", "models", "best_model_fp16.pt")
    detector_calibration: str = os.path.join(_ROOT, "checkpoints", "detector_calibration.pt")
    manifest: str = os.path.join(_ROOT, "artifacts", "models", "manifest.json")

    classifier_int8_onnx: str = os.path.join(_ROOT, "edge_deploy", "models", "classifier_int8.onnx")
    classifier_int8_mirror: str = os.path.join(_ROOT, "artifacts", "onnx_int8", "classifier_int8.onnx")
    classifier_fp32_onnx: str = os.path.join(_ROOT, "edge_deploy", "models", "classifier_fp32.onnx")


PATHS = ArtifactPaths()


def resolve_model_checkpoint(prefer_fp16_on_cuda: bool = True) -> tuple[str, str]:
    """
    Pick the best available PyTorch weights for this machine.

    Returns:
        (path, precision_label) where precision_label is 'fp16' or 'fp32'.
    """
    import torch

    use_fp16 = (
        prefer_fp16_on_cuda
        and torch.cuda.is_available()
        and os.path.isfile(PATHS.best_model_fp16)
    )
    if use_fp16:
        return PATHS.best_model_fp16, "fp16"
    if os.path.isfile(PATHS.best_model_fp32):
        return PATHS.best_model_fp32, "fp32"
    if os.path.isfile(PATHS.best_model_fp16):
        return PATHS.best_model_fp16, "fp16"
    return PATHS.best_model_fp32, "fp32"


def calibration_path() -> str:
    return PATHS.detector_calibration


def artifacts_status() -> dict:
    """Human-readable status for logs and health checks."""
    fp32 = os.path.isfile(PATHS.best_model_fp32)
    fp16 = os.path.isfile(PATHS.best_model_fp16)
    calib = os.path.isfile(PATHS.detector_calibration)
    int8 = os.path.isfile(PATHS.classifier_int8_onnx) or os.path.isfile(PATHS.classifier_int8_mirror)

    return {
        "fp32_checkpoint": fp32,
        "fp16_gpu_artifact": fp16,
        "calibration": calib,
        "onnx_int8_edge": int8,
        "gpu_recommended": "fp16" if fp16 else "fp32",
        "edge_recommended": "int8_onnx" if int8 else None,
    }
