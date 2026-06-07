"""
edge_deploy/edge_quantize.py
────────────────────────────────────────────────────────
PyTorch → ONNX FP32 → ONNX INT8 (Static PTQ) pipeline.

Handles:
  - Model wrapper for ONNX-safe tuple outputs
  - FP32 ONNX export with constant folding
  - ONNX graph simplification
  - INT8 static post-training quantization
  - FP16 fallback export
  - Calibration data generation
  - FP32 vs INT8 accuracy validation

Usage (on dev machine with GPU):
    python edge_deploy/edge_quantize.py \
        --checkpoint checkpoints/best_model.pt \
        --output edge_deploy/models/
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import cfg
from models.hybrid_model import HybridAnomalyModel


# ─────────────────────────────────────────────────────────
#  ONNX-Safe Model Wrapper
# ─────────────────────────────────────────────────────────

class OnnxExportWrapper(nn.Module):
    """Wraps HybridAnomalyModel for ONNX export.

    Changes from the original:
      1. Returns tuple instead of dict (ONNX requirement)
      2. Splits into two exportable subgraphs:
         - ClassifierModel: backbone → temporal → classifier (hot path, INT8)
         - AutoencoderModel: conv autoencoder (cold path, FP32)
    """

    def __init__(self, model: HybridAnomalyModel):
        super().__init__()
        self.backbone = model.backbone
        self.cnn_proj = model.cnn_proj
        self.temporal = model.temporal
        self.attn_pool = model.attn_pool
        self.projector = model.projector
        self.classifier = model.classifier

    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (embeddings, logits, pooled_feat)."""
        B = mel_spec.size(0)

        # CNN spatial features
        feat = self.backbone(mel_spec)
        _, C, H, W = feat.shape
        seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        seq = self.cnn_proj(seq)

        # Temporal
        seq = self.temporal(seq)

        # Attention pooling
        pooled, _ = self.attn_pool(seq)

        # Heads
        embeddings = torch.nn.functional.normalize(self.projector(pooled), dim=-1)
        logits = self.classifier(pooled)

        return embeddings, logits, pooled


class AutoencoderExportWrapper(nn.Module):
    """Wraps autoencoder branch for separate ONNX export (FP32 only)."""

    def __init__(self, model: HybridAnomalyModel):
        super().__init__()
        self.autoencoder = model.autoencoder

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        recon, _ = self.autoencoder(mel_spec)
        # Pin output shape to input shape
        if recon.shape != mel_spec.shape:
            recon = torch.nn.functional.interpolate(
                recon, size=mel_spec.shape[2:], mode="bilinear", align_corners=False
            )
        return recon


# ─────────────────────────────────────────────────────────
#  ONNX Export
# ─────────────────────────────────────────────────────────

def export_onnx_fp32(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...],
    output_names: List[str],
    opset: int = 17,
) -> str:
    """Export a model to ONNX FP32 with constant folding."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape, device=device)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"[Quantize] Exporting ONNX FP32 → {save_path}")
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={"input": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )

    # Simplify
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        model_sim, ok = onnx_simplify(onnx_model)
        if ok:
            onnx.save(model_sim, save_path)
            print("[Quantize] ONNX simplified OK")
        else:
            print("[Quantize] ONNX simplification skipped (validation failed)")
    except ImportError:
        print("[Quantize] onnx-simplifier not installed, skipping")
    except Exception as e:
        print(f"[Quantize] Simplification error (non-fatal): {e}")

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Quantize] FP32 ONNX size: {size_mb:.1f} MB")
    return save_path


# ─────────────────────────────────────────────────────────
#  FP16 Conversion
# ─────────────────────────────────────────────────────────

def convert_fp16(fp32_path: str, fp16_path: str) -> str:
    """Convert FP32 ONNX to FP16 (fallback model for RPi5)."""
    try:
        import onnx
        from onnxconverter_common import float16

        print(f"[Quantize] Converting FP16 → {fp16_path}")
        model = onnx.load(fp32_path)
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, min_positive_val=1e-7
        )
        onnx.save(model_fp16, fp16_path)
        size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
        print(f"[Quantize] FP16 ONNX size: {size_mb:.1f} MB")
        return fp16_path
    except ImportError:
        print("[Quantize] onnxconverter-common not installed, skipping FP16")
        return ""
    except Exception as e:
        print(f"[Quantize] FP16 conversion failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────
#  Calibration Data Reader
# ─────────────────────────────────────────────────────────

class MelCalibrationReader:
    """Provides calibration data for ONNX Runtime static quantization.

    Generates synthetic mel spectrograms matching the training distribution
    (normalized [0,1] range, shape 1×128×313). For best accuracy, replace
    with real samples from the MIMII normal dataset.
    """

    def __init__(
        self,
        n_samples: int = 200,
        input_shape: Tuple[int, ...] = (1, 1, 128, 313),
        data_dir: Optional[str] = None,
    ):
        self.input_shape = input_shape
        self.samples: List[np.ndarray] = []
        self.idx = 0

        if data_dir and os.path.isdir(data_dir):
            self._load_real_data(data_dir, n_samples)
        else:
            self._generate_synthetic(n_samples)

        print(f"[Quantize] Calibration set: {len(self.samples)} samples")

    def _load_real_data(self, data_dir: str, max_samples: int):
        """Load real mel spectrograms from preprocessed .npy files."""
        import glob
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))[:max_samples]
        for f in files:
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, :, :]
            elif arr.ndim == 3:
                arr = arr[np.newaxis, :, :, :]
            # Pad/trim time axis
            target_t = self.input_shape[3]
            if arr.shape[3] > target_t:
                arr = arr[:, :, :, :target_t]
            elif arr.shape[3] < target_t:
                pad = np.zeros((*arr.shape[:3], target_t - arr.shape[3]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=3)
            self.samples.append(arr)

        if not self.samples:
            print("[Quantize] No .npy files found, falling back to synthetic")
            self._generate_synthetic(200)

    def _generate_synthetic(self, n_samples: int):
        """Generate synthetic mel-like data matching training normalization."""
        rng = np.random.RandomState(42)
        for _ in range(n_samples):
            # Simulate normalized log-mel in [0, 1] with realistic distribution
            mel = rng.beta(2, 5, size=self.input_shape).astype(np.float32)
            self.samples.append(mel)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.idx >= len(self.samples):
            return None
        sample = self.samples[self.idx]
        self.idx += 1
        return {"input": sample}

    def rewind(self):
        self.idx = 0


# ─────────────────────────────────────────────────────────
#  INT8 Static PTQ
# ─────────────────────────────────────────────────────────

def quantize_int8(
    fp32_path: str,
    int8_path: str,
    calibration_reader: MelCalibrationReader,
) -> str:
    """ONNX Runtime static INT8 quantization with QDQ format.

    Key decisions:
      - QDQ format: required for ARM NEON kernels
      - MinMax calibration: stable for CNN+Transformer
      - Per-channel for Conv, per-tensor for MatMul
      - Exclude LayerNorm, Softmax, GELU from quantization
    """
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationMethod,
        QuantFormat,
        QuantType,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self, reader: MelCalibrationReader):
            self._reader = reader
        def get_next(self):
            return self._reader.get_next()
        def rewind(self):
            self._reader.rewind()

    print(f"[Quantize] INT8 static PTQ → {int8_path}")

    # Ops to keep in FP32 (quantization-sensitive)
    ops_to_exclude = []

    # Discover node names to exclude (LayerNorm, Softmax, nonlinear)
    try:
        import onnx
        model = onnx.load(fp32_path)
        for node in model.graph.node:
            if node.op_type in ("LayerNormalization", "Softmax", "Gelu",
                                "ReduceMean", "Sqrt", "Pow", "Div"):
                ops_to_exclude.append(node.name)
        print(f"[Quantize] Excluding {len(ops_to_exclude)} quantization-sensitive nodes")
    except Exception:
        pass

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=_Reader(calibration_reader),
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=ops_to_exclude,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "CalibMovingAverage": True,
            "CalibMovingAverageConstant": 0.01,
        },
    )

    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"[Quantize] INT8 ONNX size: {size_mb:.1f} MB")
    return int8_path


# ─────────────────────────────────────────────────────────
#  Calibration Data Export (for edge scoring)
# ─────────────────────────────────────────────────────────

def export_calibration(calib_pt_path: str, output_path: str) -> str:
    """Convert PyTorch calibration .pt to numpy .npz for edge deployment."""
    if not os.path.exists(calib_pt_path):
        print(f"[Quantize] No calibration at {calib_pt_path}, skipping")
        return ""

    calib = torch.load(calib_pt_path, map_location="cpu", weights_only=False)

    np_calib = {}
    for key, val in calib.items():
        if isinstance(val, torch.Tensor):
            np_calib[key] = val.numpy().astype(np.float32)
        elif isinstance(val, np.ndarray):
            np_calib[key] = val.astype(np.float32)
        elif isinstance(val, (int, float)):
            np_calib[key] = np.float32(val)

    np.savez_compressed(output_path, **np_calib)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Quantize] Calibration exported → {output_path} ({size_kb:.1f} KB)")
    return output_path


# ─────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────

def validate_quantization(fp32_path: str, int8_path: str, n_samples: int = 50):
    """Compare FP32 vs INT8 output divergence."""
    import onnxruntime as ort

    print(f"\n[Validate] Comparing FP32 vs INT8 on {n_samples} samples...")

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    input_name = sess_fp32.get_inputs()[0].name
    input_shape = sess_fp32.get_inputs()[0].shape
    # Replace dynamic dims
    shape = [s if isinstance(s, int) else 1 for s in input_shape]

    rng = np.random.RandomState(123)
    logit_diffs = []
    embed_diffs = []

    for i in range(n_samples):
        x = rng.beta(2, 5, size=shape).astype(np.float32)

        out_fp32 = sess_fp32.run(None, {input_name: x})
        out_int8 = sess_int8.run(None, {input_name: x})

        # outputs: [embeddings, logits, pooled_feat]
        embed_mae = np.mean(np.abs(out_fp32[0] - out_int8[0]))
        logit_mae = np.mean(np.abs(out_fp32[1] - out_int8[1]))
        embed_diffs.append(embed_mae)
        logit_diffs.append(logit_mae)

    print(f"  Embedding MAE: {np.mean(embed_diffs):.6f} (std: {np.std(embed_diffs):.6f})")
    print(f"  Logit MAE:     {np.mean(logit_diffs):.6f} (std: {np.std(logit_diffs):.6f})")
    print(f"  Max embed err: {np.max(embed_diffs):.6f}")
    print(f"  Max logit err: {np.max(logit_diffs):.6f}")

    if np.mean(logit_diffs) > 0.1:
        print("  ⚠ WARNING: Logit divergence > 0.1 — consider excluding more ops")
    else:
        print("  ✓ Quantization quality: ACCEPTABLE")


# ─────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────

def run_pipeline(
    checkpoint_path: str,
    output_dir: str,
    calib_data_dir: Optional[str] = None,
    n_calib_samples: int = 200,
    skip_int8: bool = False,
    skip_validation: bool = False,
):
    """Full export + quantization pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Quantize] Device: {device}")

    # Input shape: (batch, channels, n_mels, time_frames)
    # 10s audio @ 16kHz, hop=512 → 313 frames
    n_mels = cfg.data.n_mels
    n_frames = int(cfg.data.sample_rate * cfg.data.audio_duration_sec / cfg.data.hop_length) + 1
    input_shape = (1, 1, n_mels, n_frames)
    print(f"[Quantize] Input shape: {input_shape}")

    # ── Load model ────────────────────────────────────────
    print("[Quantize] Loading model...")
    model = HybridAnomalyModel(cfg.model).to(device)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"[Quantize] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Quantize] WARNING: No checkpoint at {checkpoint_path}, using random weights")

    model.eval()

    # ── Export classifier path (INT8 target) ──────────────
    classifier_wrapper = OnnxExportWrapper(model).to(device)
    classifier_wrapper.eval()

    fp32_path = os.path.join(output_dir, "classifier_fp32.onnx")
    export_onnx_fp32(
        classifier_wrapper, fp32_path, input_shape,
        output_names=["embeddings", "logits", "pooled_feat"],
    )

    # ── Export autoencoder path (FP32 only) ───────────────
    ae_wrapper = AutoencoderExportWrapper(model).to(device)
    ae_wrapper.eval()

    ae_fp32_path = os.path.join(output_dir, "autoencoder_fp32.onnx")
    export_onnx_fp32(
        ae_wrapper, ae_fp32_path, input_shape,
        output_names=["reconstruction"],
    )

    # ── FP16 fallback ─────────────────────────────────────
    fp16_path = os.path.join(output_dir, "classifier_fp16.onnx")
    convert_fp16(fp32_path, fp16_path)

    # ── INT8 quantization ─────────────────────────────────
    if not skip_int8:
        calib_reader = MelCalibrationReader(
            n_samples=n_calib_samples,
            input_shape=input_shape,
            data_dir=calib_data_dir,
        )
        int8_path = os.path.join(output_dir, "classifier_int8.onnx")
        quantize_int8(fp32_path, int8_path, calib_reader)

        if not skip_validation:
            validate_quantization(fp32_path, int8_path)

    # ── Export calibration data ────────────────────────────
    calib_pt = os.path.join(cfg.training.checkpoint_dir, "detector_calibration.pt")
    calib_npz = os.path.join(output_dir, "calibration.npz")
    export_calibration(calib_pt, calib_npz)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                print(f"  {name:40s} {size / (1024*1024):8.1f} MB")
            else:
                print(f"  {name:40s} {size / 1024:8.1f} KB")
    print("=" * 50)


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export + Quantize model for Raspberry Pi 5 deployment"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt",
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output", default="edge_deploy/models",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--calib-data", default=None,
        help="Directory of .npy calibration mel spectrograms (optional)"
    )
    parser.add_argument(
        "--n-calib", type=int, default=200,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--skip-int8", action="store_true",
        help="Skip INT8 quantization (export FP32/FP16 only)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip FP32 vs INT8 validation"
    )
    args = parser.parse_args()

    run_pipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        calib_data_dir=args.calib_data,
        n_calib_samples=args.n_calib,
        skip_int8=args.skip_int8,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
