"""
utils/validation.py
────────────────────────────────────────────────────────
Input validation and error handling utilities.

Provides:
  - Audio file validation
  - Configuration validation
  - Safe error handling with graceful fallbacks
  - Structured logging
"""

from __future__ import annotations

import os
import sys
import shutil
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Tuple
from functools import wraps
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np

from config import Config


# ─────────────────────────────────────────────────────────
#  Structured Logging Setup
# ─────────────────────────────────────────────────────────

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """Setup structured logging for production."""
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger("mimii")
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(format_str))
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = logging.getLogger("mimii")


# ─────────────────────────────────────────────────────────
#  Error Handling Decorators
# ─────────────────────────────────────────────────────────

def safe_execute(
    fallback_value: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
):
    """
    Decorator for safe execution with fallback values.

    Args:
        fallback_value: Value to return on exception
        log_errors: Whether to log exceptions
        reraise: Whether to re-raise the exception after logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {e}\n{traceback.format_exc()}"
                    )
                if reraise:
                    raise
                return fallback_value
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with automatic cleanup."""

    def __init__(
        self,
        operation_name: str,
        fallback_value: Any = None,
        cleanup: Optional[Callable] = None,
    ):
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.cleanup = cleanup
        self.error: Optional[Exception] = None

    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = exc_val
            logger.error(
                f"Operation '{self.operation_name}' failed: {exc_val}\n"
                f"{traceback.format_exc()}"
            )

        if self.cleanup:
            try:
                self.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        # Suppress exception if fallback is provided
        if exc_val and self.fallback_value is not None:
            return True

        return False


# ─────────────────────────────────────────────────────────
#  Audio File Validation
# ─────────────────────────────────────────────────────────

@dataclass
class AudioValidationResult:
    """Result of audio file validation."""
    valid: bool
    path: str
    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None
    channels: Optional[int] = None
    error_message: Optional[str] = None


def validate_audio_file(
    path: str,
    min_duration_sec: float = 0.1,
    max_duration_sec: float = 600.0,
    required_sample_rate: Optional[int] = None,
) -> AudioValidationResult:
    """
    Validate an audio file for processing.

    Args:
        path: Path to audio file
        min_duration_sec: Minimum allowed duration
        max_duration_sec: Maximum allowed duration
        required_sample_rate: If specified, validate sample rate matches

    Returns:
        AudioValidationResult with validation status and metadata
    """
    # Check file exists
    if not os.path.exists(path):
        return AudioValidationResult(
            valid=False, path=path,
            error_message=f"File not found: {path}"
        )

    # Check file is readable
    if not os.access(path, os.R_OK):
        return AudioValidationResult(
            valid=False, path=path,
            error_message=f"File not readable: {path}"
        )

    # Check file size (not empty, not too large)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 0.001:  # Less than 1KB
        return AudioValidationResult(
            valid=False, path=path,
            error_message=f"File too small ({size_mb:.3f} MB)"
        )
    if size_mb > 1024:  # Larger than 1GB
        return AudioValidationResult(
            valid=False, path=path,
            error_message=f"File too large ({size_mb:.1f} MB > 1GB limit)"
        )

    # Try to load audio
    try:
        info = torchaudio.info(path)
        sample_rate = info.sample_rate
        num_frames = info.num_frames
        channels = info.num_channels
        duration = num_frames / sample_rate

        # Validate duration
        if duration < min_duration_sec:
            return AudioValidationResult(
                valid=False, path=path,
                error_message=f"Audio too short ({duration:.2f}s < {min_duration_sec}s)"
            )
        if duration > max_duration_sec:
            return AudioValidationResult(
                valid=False, path=path,
                error_message=f"Audio too long ({duration:.2f}s > {max_duration_sec}s)"
            )

        # Validate sample rate if required
        if required_sample_rate and sample_rate != required_sample_rate:
            return AudioValidationResult(
                valid=False, path=path,
                sample_rate=sample_rate, duration_sec=duration, channels=channels,
                error_message=f"Sample rate mismatch: {sample_rate} != {required_sample_rate}"
            )

        return AudioValidationResult(
            valid=True, path=path,
            sample_rate=sample_rate, duration_sec=duration, channels=channels
        )

    except Exception as e:
        return AudioValidationResult(
            valid=False, path=path,
            error_message=f"Failed to read audio: {str(e)}"
        )


def validate_audio_batch(
    paths: List[str],
    **kwargs
) -> Tuple[List[AudioValidationResult], List[AudioValidationResult]]:
    """
    Validate multiple audio files.

    Returns:
        Tuple of (valid_results, invalid_results)
    """
    results = [validate_audio_file(p, **kwargs) for p in paths]
    valid = [r for r in results if r.valid]
    invalid = [r for r in results if not r.valid]

    if invalid:
        logger.warning(f"{len(invalid)}/{len(paths)} audio files failed validation")

    return valid, invalid


# ─────────────────────────────────────────────────────────
#  Configuration Validation
# ─────────────────────────────────────────────────────────

def validate_config(cfg: Config) -> List[str]:
    """
    Validate configuration settings.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Dataset path
    if not os.path.exists(cfg.data.dataset_dir):
        errors.append(f"Dataset directory not found: {cfg.data.dataset_dir}")

    # Audio settings
    if cfg.data.sample_rate <= 0:
        errors.append(f"Invalid sample rate: {cfg.data.sample_rate}")
    if cfg.data.audio_duration_sec <= 0:
        errors.append(f"Invalid audio duration: {cfg.data.audio_duration_sec}")
    if cfg.data.n_mels <= 0:
        errors.append(f"Invalid n_mels: {cfg.data.n_mels}")

    # Training settings
    if cfg.training.batch_size <= 0:
        errors.append(f"Invalid batch size: {cfg.training.batch_size}")
    if cfg.training.epochs <= 0:
        errors.append(f"Invalid epochs: {cfg.training.epochs}")
    if cfg.training.learning_rate <= 0:
        errors.append(f"Invalid learning rate: {cfg.training.learning_rate}")

    # Model settings
    valid_backbones = ["efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "resnet50"]
    if cfg.model.backbone not in valid_backbones:
        errors.append(f"Invalid backbone: {cfg.model.backbone}. Choose from {valid_backbones}")

    valid_temporal = ["transformer", "bilstm"]
    if cfg.model.temporal_module not in valid_temporal:
        errors.append(f"Invalid temporal module: {cfg.model.temporal_module}")

    # Inference weights should sum to 1.0
    weight_sum = cfg.inference.w_recon + cfg.inference.w_embed + cfg.inference.w_mahal + cfg.inference.w_contra
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Inference weights don't sum to 1.0: {weight_sum:.3f}")

    # GPU availability
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA requested but not available")

    return errors


# ─────────────────────────────────────────────────────────
#  Safe Audio Loading
# ─────────────────────────────────────────────────────────

def safe_load_audio(
    path: str,
    target_sample_rate: int,
    target_length: int,
    mono: bool = True,
) -> Optional[torch.Tensor]:
    """
    Safely load and preprocess audio with error handling.

    Returns:
        Preprocessed waveform tensor or None if failed
    """
    try:
        # Validate first
        validation = validate_audio_file(path, required_sample_rate=None)
        if not validation.valid:
            logger.error(f"Audio validation failed: {validation.error_message}")
            return None

        # Load
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=target_sample_rate
            )

        # Normalize
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform / peak

        # Pad/trim
        length = waveform.shape[-1]
        if length > target_length:
            waveform = waveform[..., :target_length]
        elif length < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - length)
            )

        return waveform

    except Exception as e:
        logger.error(f"Failed to load audio {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────
#  Health Checks
# ─────────────────────────────────────────────────────────

def system_health_check() -> Dict[str, Any]:
    """
    Perform system health check.

    Returns:
        Dictionary with health status information
    """
    status = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "status": "healthy",
        "issues": [],
    }

    if torch.cuda.is_available():
        try:
            status["cuda_version"] = torch.version.cuda
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Test CUDA operation
            test_tensor = torch.randn(100, 100).cuda()
            _ = test_tensor @ test_tensor.T
            torch.cuda.synchronize()

        except Exception as e:
            status["status"] = "degraded"
            status["issues"].append(f"GPU error: {e}")
    else:
        status["issues"].append("CUDA not available (CPU mode)")

    # Check disk space (cross-platform)
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / 1e9
        status["disk_free_gb"] = round(free_gb, 2)
        if free_gb < 1.0:
            status["status"] = "critical"
            status["issues"].append(f"Low disk space: {free_gb:.1f}GB")
    except Exception as e:
        status["issues"].append(f"Disk check failed: {e}")

    return status
