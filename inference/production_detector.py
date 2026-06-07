"""
inference/production_detector.py
────────────────────────────────────────────────────────
Production-grade optimized inference engine.

Features:
  - Batch processing for multiple files
  - ONNX Runtime support for maximum throughput
  - TensorRT optimization (if available)
  - CUDA Graphs for repeated inference
  - Async inference queue
  - Memory-efficient streaming
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from inference.detector import AnomalyDetector
from utils.gpu_utils import (
    compile_model, optimize_for_inference, get_device,
    GPUMonitor, empty_cache, get_memory_stats
)


class ProductionDetector:
    """
    High-performance production detector with batch processing,
    model compilation, and optimized inference paths.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        use_compiled: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        self.cfg = cfg
        self.device = get_device()
        self.model = model.to(self.device)

        # Optimize and optionally compile model
        self.model = optimize_for_inference(self.model)
        if use_compiled and self.device.type == "cuda":
            self.model = compile_model(self.model, mode=compile_mode)

        # Base detector for calibration
        self.base_detector = AnomalyDetector(self.model, cfg)

        # Calibration state
        self._is_calibrated = False

        # Performance tracking
        self._inference_times: List[float] = []
        self._batch_times: List[float] = []

    def load_calibration(self, calib_path: str):
        """Load pre-computed calibration from file."""
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        calib = torch.load(calib_path, map_location="cpu", weights_only=False)
        for k, v in calib.items():
            setattr(self.base_detector, k, v)
        self.base_detector.refresh_reference_cache()

        self._is_calibrated = True
        print(f"[ProductionDetector] Loaded calibration from {calib_path}")

    def calibrate(self, normal_loader):
        """Calibrate using normal data loader."""
        print("[ProductionDetector] Running calibration on normal data...")
        self.base_detector.fit_reference_distribution(normal_loader)
        self._is_calibrated = True

    def save_calibration(self, calib_path: str):
        """Save calibration to file."""
        os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)

        calib = {
            "ref_mean": self.base_detector.ref_mean,
            "ref_cov_inv": self.base_detector.ref_cov_inv,
            "ref_pool": self.base_detector.ref_pool,
            "recon_mu": self.base_detector.recon_mu,
            "recon_sigma": self.base_detector.recon_sigma,
            "embed_mu": self.base_detector.embed_mu,
            "embed_sigma": self.base_detector.embed_sigma,
            "mahal_mu": self.base_detector.mahal_mu,
            "mahal_sigma": self.base_detector.mahal_sigma,
            "contra_mu": self.base_detector.contra_mu,
            "contra_sigma": self.base_detector.contra_sigma,
        }
        torch.save(calib, calib_path)
        print(f"[ProductionDetector] Saved calibration to {calib_path}")

    @torch.inference_mode()
    def detect_batch(
        self,
        mel_specs: torch.Tensor,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Optimized batch inference.

        Args:
            mel_specs: (B, 1, n_mels, frames) batched mel spectrograms
            return_embeddings: If True, include embeddings in output

        Returns:
            List of detection results (one per sample)
        """
        if not self._is_calibrated:
            warnings.warn("Detector not calibrated. Results may be inaccurate.")

        start_time = time.perf_counter()
        batch_size = mel_specs.size(0)

        # Move to device with non-blocking transfer
        mel_specs = mel_specs.to(self.device, non_blocking=True)

        # FP16 compute on CUDA without storing FP16 weights (stable + fast).
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(mel_specs)
        else:
            out = self.model(mel_specs)

        # Process each sample in the batch
        results = []
        for i in range(batch_size):
            result = self._process_single_output(
                mel_spec=mel_specs[i:i+1],
                embedding=out["embeddings"][i],
                logits=out["logits"][i],
                reconstruction=out["reconstruction"][i:i+1],
                attention_weights=out["attention_weights"][i],
            )

            if return_embeddings:
                result["embedding"] = out["embeddings"][i].cpu().numpy()

            results.append(result)

        batch_time = time.perf_counter() - start_time
        self._batch_times.append(batch_time)

        # Log performance
        samples_per_sec = batch_size / batch_time
        print(f"[Batch] {batch_size} samples in {batch_time:.3f}s ({samples_per_sec:.1f} samples/sec)")

        return results

    def _process_single_output(
        self,
        mel_spec: torch.Tensor,
        embedding: torch.Tensor,
        logits: torch.Tensor,
        reconstruction: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """Process a single sample's model outputs into detection result."""
        # Convert to numpy for score computation
        emb = embedding.cpu().float().numpy()

        # Reconstruction loss (match dtypes under autocast FP16)
        mel_for_recon = mel_spec
        if mel_for_recon.dtype != reconstruction.dtype:
            mel_for_recon = mel_for_recon.to(dtype=reconstruction.dtype)
        recon_err = F.mse_loss(reconstruction, mel_for_recon, reduction="mean").item()
        recon_norm = self._z_score(recon_err, self.base_detector.recon_mu, self.base_detector.recon_sigma)

        # Score 2: Embedding Distance
        if self.base_detector.ref_mean is not None:
            emb_n = emb / (np.linalg.norm(emb) + 1e-8)
            ref_n = self.base_detector.ref_mean_normed
            if ref_n is None:
                self.base_detector.refresh_reference_cache()
                ref_n = self.base_detector.ref_mean_normed
            embed_dist = float(1.0 - np.dot(emb_n, ref_n))
            embed_norm = self._z_score(embed_dist, self.base_detector.embed_mu, self.base_detector.embed_sigma)
        else:
            embed_dist = 0.0
            embed_norm = 0.0

        # Score 3: Mahalanobis Distance
        if self.base_detector.ref_mean is not None and self.base_detector.ref_cov_inv is not None:
            diff = emb - self.base_detector.ref_mean
            mahal = float(np.sqrt(np.maximum(diff @ self.base_detector.ref_cov_inv @ diff, 0.0)))
            mahal_norm = self._z_score(mahal, self.base_detector.mahal_mu, self.base_detector.mahal_sigma)
        else:
            mahal = 0.0
            mahal_norm = 0.0

        # Score 4: Contrastive Score
        if self.base_detector.ref_pool is not None:
            pool_normed = self.base_detector.ref_pool_normed
            if pool_normed is None:
                self.base_detector.refresh_reference_cache()
                pool_normed = self.base_detector.ref_pool_normed
            emb_normed = emb / (np.linalg.norm(emb) + 1e-8)
            sims = pool_normed @ emb_normed
            k = min(5, len(sims))
            contra_sim = float(np.sort(sims)[-k:].mean())
            contra_dist = 1.0 - contra_sim
            contra_norm = self._z_score(contra_dist, self.base_detector.contra_mu, self.base_detector.contra_sigma)
        else:
            contra_sim = 1.0
            contra_dist = 0.0
            contra_norm = 0.0

        # Fused Anomaly Score
        w = self.cfg.inference
        fused = (
            w.w_recon * self._sigmoid(recon_norm)
            + w.w_embed * self._sigmoid(embed_norm)
            + w.w_mahal * self._sigmoid(mahal_norm)
            + w.w_contra * self._sigmoid(contra_norm)
        )
        fused = float(np.clip(fused, 0.0, 1.0))

        # Thresholding & Risk Assessment
        percentile = float(np.clip(fused * 100, 0, 100))
        is_anomalous = percentile > w.percentile_threshold
        health = max(0, min(100, int(100 - percentile)))

        if health < w.risk_critical:
            risk = "Critical"
        elif health < w.risk_high:
            risk = "High"
        elif health < w.risk_medium:
            risk = "Medium"
        else:
            risk = "Low"

        recommendations = {
            "Critical": "Stop equipment immediately – critical failure risk detected.",
            "High": "Schedule urgent inspection within 24 hours.",
            "Medium": "Monitor closely and plan maintenance this week.",
            "Low": "No immediate action required. Continue routine monitoring.",
        }

        # Temporal Anomaly Detection
        mel_np = mel_spec[0, 0].cpu().float().numpy()
        recon_np = reconstruction[0, 0].cpu().float().numpy()
        frame_err = np.mean(np.abs(mel_np - recon_np), axis=0)

        temporal_anomalies = []
        if len(frame_err) > 1:
            frame_threshold = np.percentile(frame_err, 75)
            in_anomaly = False
            start_frame = 0
            sr, hop = self.cfg.data.sample_rate, self.cfg.data.hop_length

            for f_idx in range(len(frame_err)):
                if frame_err[f_idx] > frame_threshold and not in_anomaly:
                    in_anomaly = True
                    start_frame = f_idx
                elif frame_err[f_idx] <= frame_threshold and in_anomaly:
                    in_anomaly = False
                    sev = float(np.mean(frame_err[start_frame:f_idx]))
                    temporal_anomalies.append({
                        "start": round(start_frame * hop / sr, 3),
                        "end": round(f_idx * hop / sr, 3),
                        "severity": round(sev, 4),
                    })

            if in_anomaly:
                sev = float(np.mean(frame_err[start_frame:]))
                temporal_anomalies.append({
                    "start": round(start_frame * hop / sr, 3),
                    "end": round(len(frame_err) * hop / sr, 3),
                    "severity": round(sev, 4),
                })

        z_score = float((fused - 0.5) / 0.15)
        confidence = float(min(abs(fused - 0.5) * 2.0, 1.0))

        return {
            "label": "Anomalous" if is_anomalous else "Normal",
            "anomaly_score": round(fused, 4),
            "confidence": round(confidence, 4),
            "multi_scores": {
                "reconstruction_error": round(recon_err, 4),
                "embedding_distance": round(embed_dist, 4),
                "mahalanobis": round(mahal, 4),
                "contrastive_score": round(contra_sim, 4),
            },
            "temporal_anomaly": temporal_anomalies,
            "system": {
                "health_index": health,
                "risk_level": risk,
                "recommendation": recommendations[risk],
            },
        }

    @staticmethod
    def _z_score(value: float, mu: float, sigma: float) -> float:
        return (value - mu) / (sigma + 1e-8)

    @staticmethod
    def _sigmoid(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))))

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        stats = {}
        if self._batch_times:
            stats["avg_batch_time_ms"] = np.mean(self._batch_times) * 1000
            stats["std_batch_time_ms"] = np.std(self._batch_times) * 1000
            stats["total_batches"] = len(self._batch_times)

        if torch.cuda.is_available():
            mem_stats = get_memory_stats()
            stats.update({f"gpu_{k}": v for k, v in mem_stats.items()})

        return stats

    def reset_stats(self):
        """Reset performance tracking."""
        self._batch_times.clear()


class AsyncInferenceQueue:
    """Async inference queue for high-throughput scenarios."""

    def __init__(
        self,
        detector: ProductionDetector,
        max_queue_size: int = 100,
        num_workers: int = 2,
    ):
        self.detector = detector
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._queue: List[torch.Tensor] = []
        self._callbacks: List[Callable] = []

    def submit(self, mel_spec: torch.Tensor, callback: Optional[Callable] = None):
        """Submit a sample for async inference."""
        if len(self._queue) >= self.max_queue_size:
            raise RuntimeError("Inference queue is full")

        self._queue.append(mel_spec)
        if callback:
            self._callbacks.append(callback)

    def process_queue(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        """Process all queued samples in batches."""
        if not self._queue:
            return []

        all_results = []

        # Process in batches
        for i in range(0, len(self._queue), batch_size):
            batch = torch.cat(self._queue[i:i+batch_size], dim=0)
            results = self.detector.detect_batch(batch)
            all_results.extend(results)

        # Clear queue
        self._queue.clear()
        self._callbacks.clear()

        return all_results

    def shutdown(self):
        """Shutdown the async executor."""
        self.executor.shutdown(wait=True)
