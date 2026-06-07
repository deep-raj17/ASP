"""
production_api.py
────────────────────────────────────────────────────────
Production API server for batch inference.

Features:
  - REST API for single and batch inference
  - Async job queue for large batches
  - Health check endpoint
  - Metrics endpoint
  - GPU-optimized inference
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import tempfile

import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from paths import resolve_model_checkpoint, calibration_path
from models.hybrid_model import HybridAnomalyModel
from inference.production_detector import ProductionDetector
from utils.audio_utils import AudioProcessor, pad_or_trim
from utils.gpu_utils import setup_cuda_optimizations, get_memory_stats
from utils.checkpoint import load_model_weights
from utils.validation import (
    validate_audio_file, safe_load_audio, setup_logging,
    system_health_check, logger
)

# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────

app = Flask(__name__)

# Global detector instance
DETECTOR: Optional[ProductionDetector] = None
PROCESSOR: Optional[AudioProcessor] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LEN = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)

# Thread pool for async processing
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Job tracking for async batches
JOBS: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────
#  Initialization
# ─────────────────────────────────────────────────────────

def initialize():
    """Initialize the production system."""
    global DETECTOR, PROCESSOR

    setup_logging(log_file="logs/production_api.log")
    setup_cuda_optimizations()

    # Load model
    model = HybridAnomalyModel(cfg.model).to(DEVICE)

    ckpt_path, _ = resolve_model_checkpoint(prefer_fp16_on_cuda=True)
    if os.path.exists(ckpt_path):
        epoch, loaded_prec = load_model_weights(model, ckpt_path, DEVICE)
        logger.info(f"Loaded checkpoint ({loaded_prec}) from {ckpt_path} epoch={epoch}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}")

    use_compile = os.environ.get("APP_USE_TORCH_COMPILE", "0") == "1"
    DETECTOR = ProductionDetector(model, cfg, use_compiled=use_compile)

    calib_path = calibration_path()
    if os.path.exists(calib_path):
        DETECTOR.load_calibration(calib_path)
    else:
        logger.warning(f"No calibration found at {calib_path}")

    PROCESSOR = AudioProcessor(cfg.data)

    logger.info("Production API initialized successfully")


# ─────────────────────────────────────────────────────────
#  Enhanced Output Generator
# ─────────────────────────────────────────────────────────

def generate_output(detector_result: dict, native_sr: int, audio_duration: float) -> Dict[str, Any]:
    """Build the enriched diagnostic output with all requested fields.

    Takes the raw detector result and augments it with:
      - machine type
      - status (normal/abnormal)
      - confidence
      - severity (low/medium/high)
      - fault_type
      - anomaly_score
      - recommendation
    Plus the full multi-score breakdown, temporal anomalies, and system health.
    """
    anomaly_score = detector_result["anomaly_score"]
    confidence    = detector_result["confidence"]
    is_anomalous  = detector_result["label"] == "Anomalous"

    # Severity logic based on anomaly_score
    if anomaly_score > 0.8:
        severity = "high"
    elif anomaly_score > 0.5:
        severity = "medium"
    else:
        severity = "low"

    # Fault type classification (heuristic from score profile)
    multi = detector_result.get("multi_scores", {})
    recon_err = multi.get("reconstruction_error", 0)
    mahal_dist = multi.get("mahalanobis", 0)

    if anomaly_score > 0.7:
        if recon_err > mahal_dist:
            fault_type = "bearing_fault"
        else:
            fault_type = "mechanical_looseness"
    elif anomaly_score > 0.5:
        fault_type = "early_degradation"
    else:
        fault_type = "normal_variation"

    # Recommendation based on severity
    if severity == "high":
        recommendation = "Immediate inspection required. Stop equipment if possible."
    elif severity == "medium":
        recommendation = "Schedule maintenance soon. Monitor closely."
    else:
        recommendation = "System operating normally. Continue routine monitoring."

    return {
        # ── Core Fields ────────────────────────────────────
        "machine": "fan",
        "status": "abnormal" if is_anomalous else "normal",
        "confidence": round(confidence, 3),
        "severity": severity,
        "fault_type": fault_type,
        "anomaly_score": round(anomaly_score, 3),
        "recommendation": recommendation,

        # ── Detailed Diagnostics ──────────────────────────
        "multi_scores": detector_result.get("multi_scores", {}),
        "temporal_anomaly": detector_result.get("temporal_anomaly", []),

        # ── System Health ─────────────────────────────────
        "system": detector_result.get("system", {}),

        # ── Audio Metadata ────────────────────────────────
        "audio_info": {
            "native_sample_rate_hz": native_sr,
            "processing_sample_rate_hz": cfg.data.sample_rate,
            "duration_sec": round(audio_duration, 2),
            "resampled": native_sr != cfg.data.sample_rate,
        },

        # ── Model Metadata ────────────────────────────────
        "model_info": {
            "backbone": cfg.model.backbone,
            "temporal_module": cfg.model.temporal_module,
            "device": str(DEVICE),
            "n_mels": cfg.data.n_mels,
            "n_mfcc": cfg.data.n_mfcc,
            "fmin_hz": cfg.data.fmin,
            "fmax_hz": cfg.data.fmax,
        },
    }


# ─────────────────────────────────────────────────────────
#  Inference Functions
# ─────────────────────────────────────────────────────────

def process_single_file(file_path: str) -> Dict[str, Any]:
    """Process a single audio file and return enriched diagnostic output."""
    # Validate (do NOT enforce sample rate — we resample any rate)
    validation = validate_audio_file(file_path)
    if not validation.valid:
        return {"error": validation.error_message, "file": file_path}

    # Load audio at its NATIVE sample rate first
    try:
        waveform, native_sr = torchaudio.load(file_path)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return {"error": f"Failed to load audio file: {e}", "file": file_path}

    # Compute duration from native rate before resampling
    audio_duration = waveform.shape[-1] / native_sr

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to processing sample rate (handles 40kHz, 44.1kHz, 48kHz etc.)
    if native_sr != cfg.data.sample_rate:
        logger.info(f"Resampling: {native_sr}Hz → {cfg.data.sample_rate}Hz")
        waveform = torchaudio.functional.resample(waveform, native_sr, cfg.data.sample_rate)

    waveform = pad_or_trim(waveform, TARGET_LEN)

    # Extract features
    mel, _, _ = PROCESSOR(waveform, augment=False)
    mel_batch = mel.unsqueeze(0).to(DEVICE)

    # Run detection
    results = DETECTOR.detect_batch(mel_batch)
    raw_result = results[0]

    # Build enriched output
    enriched = generate_output(raw_result, native_sr, audio_duration)

    return {
        "file": file_path,
        "result": enriched,
    }


def process_batch(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple files and return enriched diagnostic output."""
    results = []

    # Validate all files first (do NOT enforce sample rate — we resample any rate)
    valid_paths = []
    for path in file_paths:
        validation = validate_audio_file(path)
        if validation.valid:
            valid_paths.append(path)
        else:
            results.append({
                "file": path,
                "error": validation.error_message,
            })

    # Batch inference with per-file resampling
    if valid_paths:
        mels = []
        audio_metadata = []  # Track native SR and duration per file
        processed_paths = []

        for path in valid_paths:
            try:
                waveform, native_sr = torchaudio.load(path)
                audio_duration = waveform.shape[-1] / native_sr

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample to processing sample rate
                if native_sr != cfg.data.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, native_sr, cfg.data.sample_rate
                    )

                waveform = pad_or_trim(waveform, TARGET_LEN)
                mel, _, _ = PROCESSOR(waveform, augment=False)
                mels.append(mel)
                audio_metadata.append((native_sr, audio_duration))
                processed_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({"file": path, "error": str(e)})

        if mels:
            batch = torch.stack(mels).to(DEVICE)
            batch_results = DETECTOR.detect_batch(batch)

            for path, raw_result, (native_sr, duration) in zip(
                processed_paths[:len(batch_results)],
                batch_results,
                audio_metadata[:len(batch_results)],
            ):
                enriched = generate_output(raw_result, native_sr, duration)
                results.append({
                    "file": path,
                    "result": enriched,
                })

    return results


# ─────────────────────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    status = system_health_check()
    code = 200 if status["status"] == "healthy" else 503
    return jsonify(status), code


@app.route("/predict", methods=["POST"])
def predict():
    """Single file prediction endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Save to temp file (cross-platform)
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    file.save(temp_path)

    try:
        result = process_single_file(temp_path)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint (synchronous)."""
    data = request.get_json()
    if not data or "files" not in data:
        return jsonify({"error": "No files provided"}), 400

    file_paths = data["files"]
    if not isinstance(file_paths, list):
        return jsonify({"error": "files must be a list"}), 400

    if len(file_paths) > 100:
        return jsonify({"error": "Batch size limited to 100 files"}), 400

    try:
        results = process_batch(file_paths)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/async", methods=["POST"])
def predict_async():
    """Async batch prediction endpoint."""
    data = request.get_json()
    if not data or "files" not in data:
        return jsonify({"error": "No files provided"}), 400

    file_paths = data["files"]
    job_id = f"job_{int(time.time() * 1000)}"

    # Submit async job
    JOBS[job_id] = {
        "status": "pending",
        "files": file_paths,
        "results": None,
        "submitted_at": time.time(),
    }

    def run_job():
        JOBS[job_id]["status"] = "running"
        try:
            results = process_batch(file_paths)
            JOBS[job_id]["results"] = results
            JOBS[job_id]["status"] = "completed"
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["completed_at"] = time.time()

    EXECUTOR.submit(run_job)

    return jsonify({
        "job_id": job_id,
        "status": "pending",
        "file_count": len(file_paths),
    })


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get async job status."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404

    job = JOBS[job_id].copy()

    # Calculate duration
    if job.get("completed_at"):
        job["duration_sec"] = job["completed_at"] - job["submitted_at"]

    return jsonify(job)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get system metrics."""
    metrics = {
        "gpu": get_memory_stats() if torch.cuda.is_available() else None,
        "inference": DETECTOR.get_performance_stats() if DETECTOR else None,
        "active_jobs": len([j for j in JOBS.values() if j["status"] == "running"]),
    }
    return jsonify(metrics)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    initialize()

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        EXECUTOR.shutdown(wait=True)
