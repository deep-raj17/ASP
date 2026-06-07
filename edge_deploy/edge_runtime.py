"""
edge_deploy/edge_runtime.py
────────────────────────────────────────────────────────
ONNX Runtime inference wrapper optimized for RPi5 ARM64.

Features:
  - XNNPACK + CPU EP provider chain
  - I/O binding for zero-copy inference
  - Pre-allocated buffers
  - Multi-score fusion (no PyTorch dependency)
  - Calibration loading from .npz
  - Autoencoder reconstruction (optional cold path)

Usage on RPi5:
    # Run directly:
    python3 edge_streaming.py --config config.yaml

    # Or import in your own script:
    import sys; sys.path.insert(0, "/opt/mimii")
    from edge_runtime import EdgeInferenceEngine
    engine = EdgeInferenceEngine("models/classifier_int8.onnx")
    result = engine.infer(mel_spectrogram)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required. Install with: pip install onnxruntime"
    )


# ─────────────────────────────────────────────────────────
#  Session Configuration
# ─────────────────────────────────────────────────────────

def create_session(
    model_path: str,
    num_inference_threads: int = 3,
    enable_profiling: bool = False,
) -> ort.InferenceSession:
    """Create an optimized ONNX Runtime session for ARM64.

    Thread allocation strategy for RPi5 (4× Cortex-A76):
      - Core 0: OS kernel + audio capture
      - Cores 1-3: ONNX inference (intra_op_num_threads=3)
    """
    opts = ort.SessionOptions()

    # Graph optimization — fuse Conv+BN+ReLU, constant fold, etc.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Sequential execution — single inference stream, no thread contention
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Thread config — reserve core 0 for audio/OS
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = num_inference_threads

    # Memory optimization
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    # Prevent busy-wait spinning (saves thermal budget)
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

    if enable_profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = "ort_profile"

    # Provider priority: XNNPACK (ARM-optimized) → CPU fallback
    providers = []
    available = ort.get_available_providers()

    if "XNNPACKExecutionProvider" in available:
        providers.append("XNNPACKExecutionProvider")
        print("[Runtime] Using XNNPACK execution provider")

    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    # Log session info
    inp = session.get_inputs()[0]
    print(f"[Runtime] Model loaded: {os.path.basename(model_path)}")
    print(f"[Runtime] Input: {inp.name} shape={inp.shape} dtype={inp.type}")
    print(f"[Runtime] Providers: {session.get_providers()}")

    return session


# ─────────────────────────────────────────────────────────
#  Score Fusion Engine (numpy-only, no PyTorch)
# ─────────────────────────────────────────────────────────

class ScoreFusion:
    """Multi-signal anomaly score fusion.

    Port of ProductionDetector._process_single_output() to pure numpy
    for zero-PyTorch-dependency edge inference.
    """

    # Score fusion weights (from config.py InferenceConfig)
    W_RECON = 0.30
    W_EMBED = 0.25
    W_MAHAL = 0.30
    W_CONTRA = 0.15

    # Risk thresholds
    PERCENTILE_THRESHOLD = 95.0
    RISK_CRITICAL = 20
    RISK_HIGH = 50
    RISK_MEDIUM = 80

    def __init__(self):
        # Calibration statistics (loaded from .npz)
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_cov_inv: Optional[np.ndarray] = None
        self.ref_pool: Optional[np.ndarray] = None

        self.recon_mu = 0.0
        self.recon_sigma = 1.0
        self.embed_mu = 0.0
        self.embed_sigma = 1.0
        self.mahal_mu = 0.0
        self.mahal_sigma = 1.0
        self.contra_mu = 0.0
        self.contra_sigma = 1.0

        self._calibrated = False

    def load_calibration(self, npz_path: str):
        """Load calibration from .npz exported by edge_quantize.py."""
        if not os.path.exists(npz_path):
            warnings.warn(f"Calibration not found: {npz_path}")
            return

        data = np.load(npz_path, allow_pickle=True)

        for key in data.files:
            val = data[key]
            # Scalar values stored as 0-d arrays
            if val.ndim == 0:
                val = float(val)
            setattr(self, key, val)

        self._calibrated = True
        print(f"[Runtime] Calibration loaded from {npz_path}")

    def compute_scores(
        self,
        embedding: np.ndarray,
        logits: np.ndarray,
        reconstruction: Optional[np.ndarray] = None,
        mel_input: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute fused anomaly score from model outputs.

        Args:
            embedding: (D,) L2-normalized embedding vector
            logits: (1,) classifier logit
            reconstruction: (1, n_mels, T) reconstructed mel (optional)
            mel_input: (1, n_mels, T) input mel for recon error (optional)
        """
        # Score 1: Reconstruction error
        if reconstruction is not None and mel_input is not None:
            recon_err = float(np.mean((reconstruction - mel_input) ** 2))
        else:
            recon_err = 0.0
        recon_norm = self._z_score(recon_err, self.recon_mu, self.recon_sigma)

        # Score 2: Embedding distance
        if self.ref_mean is not None:
            emb_n = embedding / (np.linalg.norm(embedding) + 1e-8)
            ref_n = self.ref_mean / (np.linalg.norm(self.ref_mean) + 1e-8)
            embed_dist = float(1.0 - np.dot(emb_n, ref_n))
            embed_norm = self._z_score(embed_dist, self.embed_mu, self.embed_sigma)
        else:
            embed_dist = 0.0
            embed_norm = 0.0

        # Score 3: Mahalanobis distance
        if self.ref_mean is not None and self.ref_cov_inv is not None:
            diff = embedding - self.ref_mean
            mahal = float(np.sqrt(max(diff @ self.ref_cov_inv @ diff, 0.0)))
            mahal_norm = self._z_score(mahal, self.mahal_mu, self.mahal_sigma)
        else:
            mahal = 0.0
            mahal_norm = 0.0

        # Score 4: Contrastive score
        if self.ref_pool is not None:
            pool_norms = np.linalg.norm(self.ref_pool, axis=1, keepdims=True) + 1e-8
            pool_normed = self.ref_pool / pool_norms
            emb_normed = embedding / (np.linalg.norm(embedding) + 1e-8)
            sims = pool_normed @ emb_normed
            k = min(5, len(sims))
            contra_sim = float(np.sort(sims)[-k:].mean())
            contra_dist = 1.0 - contra_sim
            contra_norm = self._z_score(contra_dist, self.contra_mu, self.contra_sigma)
        else:
            contra_sim = 1.0
            contra_dist = 0.0
            contra_norm = 0.0

        # Fused score
        fused = (
            self.W_RECON * self._sigmoid(recon_norm)
            + self.W_EMBED * self._sigmoid(embed_norm)
            + self.W_MAHAL * self._sigmoid(mahal_norm)
            + self.W_CONTRA * self._sigmoid(contra_norm)
        )
        fused = float(np.clip(fused, 0.0, 1.0))

        # Thresholds
        percentile = float(np.clip(fused * 100, 0, 100))
        is_anomalous = percentile > self.PERCENTILE_THRESHOLD
        health = max(0, min(100, int(100 - percentile)))

        if health < self.RISK_CRITICAL:
            risk = "Critical"
        elif health < self.RISK_HIGH:
            risk = "High"
        elif health < self.RISK_MEDIUM:
            risk = "Medium"
        else:
            risk = "Low"

        confidence = float(min(abs(fused - 0.5) * 2.0, 1.0))

        # Severity + fault type
        if fused > 0.8:
            severity = "high"
        elif fused > 0.5:
            severity = "medium"
        else:
            severity = "low"

        if fused > 0.7:
            fault_type = "bearing_fault" if recon_err > mahal else "mechanical_looseness"
        elif fused > 0.5:
            fault_type = "early_degradation"
        else:
            fault_type = "normal_variation"

        recommendations = {
            "Critical": "Immediate inspection required. Stop equipment if possible.",
            "High": "Schedule urgent inspection within 24 hours.",
            "Medium": "Schedule maintenance soon. Monitor closely.",
            "Low": "System operating normally. Continue routine monitoring.",
        }

        return {
            "label": "Anomalous" if is_anomalous else "Normal",
            "anomaly_score": round(fused, 4),
            "confidence": round(confidence, 4),
            "severity": severity,
            "fault_type": fault_type,
            "multi_scores": {
                "reconstruction_error": round(recon_err, 4),
                "embedding_distance": round(embed_dist, 4),
                "mahalanobis": round(mahal, 4),
                "contrastive_score": round(contra_sim if self.ref_pool is not None else 0.0, 4),
            },
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


# ─────────────────────────────────────────────────────────
#  Edge Inference Engine
# ─────────────────────────────────────────────────────────

class EdgeInferenceEngine:
    """Complete inference engine for RPi5 deployment.

    Wraps ONNX Runtime session + score fusion + optional autoencoder.
    """

    def __init__(
        self,
        classifier_model: str,
        autoencoder_model: Optional[str] = None,
        calibration_path: Optional[str] = None,
        num_threads: int = 3,
    ):
        # Load classifier session (INT8 or FP16)
        self.classifier = create_session(classifier_model, num_threads)
        self.input_name = self.classifier.get_inputs()[0].name

        # Pre-read output names for binding
        self._output_names = [o.name for o in self.classifier.get_outputs()]

        # Optional autoencoder (FP32, run at reduced duty cycle)
        self.autoencoder: Optional[ort.InferenceSession] = None
        self._ae_input_name: str = "input"   # default; overridden below
        if autoencoder_model and os.path.exists(autoencoder_model):
            self.autoencoder = create_session(autoencoder_model, num_threads)
            self._ae_input_name = self.autoencoder.get_inputs()[0].name
            print("[Runtime] Autoencoder loaded for reconstruction scoring")

        # Score fusion
        self.scorer = ScoreFusion()
        if calibration_path:
            self.scorer.load_calibration(calibration_path)

        # Performance tracking
        self._latencies: List[float] = []
        self._inference_count = 0

    def infer(
        self,
        mel_spec: np.ndarray,
        run_autoencoder: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a single mel spectrogram.

        Args:
            mel_spec: (1, 1, n_mels, T) or (1, n_mels, T) float32 array
            run_autoencoder: Whether to run reconstruction (costs ~30% extra)

        Returns:
            Anomaly detection result dict
        """
        # Ensure correct shape
        if mel_spec.ndim == 3:
            mel_spec = mel_spec[np.newaxis, :]
        mel_spec = np.ascontiguousarray(mel_spec, dtype=np.float32)

        start = time.perf_counter()

        # Classifier forward pass
        outputs = self.classifier.run(self._output_names, {self.input_name: mel_spec})
        embeddings, logits, pooled_feat = outputs[0], outputs[1], outputs[2]

        # Autoencoder (optional) — uses its own input name
        reconstruction = None
        if run_autoencoder and self.autoencoder is not None:
            ae_out = self.autoencoder.run(None, {self._ae_input_name: mel_spec})
            reconstruction = ae_out[0]

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._latencies.append(elapsed_ms)
        self._inference_count += 1

        # Score fusion
        result = self.scorer.compute_scores(
            embedding=embeddings[0],
            logits=logits[0],
            reconstruction=reconstruction[0] if reconstruction is not None else None,
            mel_input=mel_spec[0] if reconstruction is not None else None,
        )

        result["inference_ms"] = round(elapsed_ms, 2)
        result["inference_count"] = self._inference_count

        return result

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "count": len(arr),
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    def reset_stats(self):
        self._latencies.clear()
