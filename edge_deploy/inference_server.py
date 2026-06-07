"""
edge_deploy/inference_server.py
────────────────────────────────────────────────────────
FastAPI HTTP inference server for Raspberry Pi 5.

Uses EdgeInferenceEngine (full multi-score fusion + calibration),
not raw logits only.

Usage:
    cd /opt/mimii
    source venv/bin/activate
    uvicorn inference_server:app --host 0.0.0.0 --port 8000

    # Or via systemd:
    sudo systemctl start mimii-api
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

EDGE_DIR = Path(__file__).resolve().parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from edge_runtime import EdgeInferenceEngine
from edge_streaming import MelTransform, mel_time_frames

DEFAULT_CONFIG = EDGE_DIR / "config.yaml"
N_MELS = 128


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path or os.environ.get("MIMII_CONFIG", DEFAULT_CONFIG))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _resolve_model_path(cfg: dict, key: str) -> str:
    raw = cfg.get(key, "")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(os.environ.get("MIMII_ROOT", Path.cwd())) / p
    return str(p)


cfg = load_config()
engine = EdgeInferenceEngine(
    classifier_model=_resolve_model_path(cfg, "classifier_model"),
    autoencoder_model=_resolve_model_path(cfg, "autoencoder_model"),
    calibration_path=_resolve_model_path(cfg, "calibration_path"),
    num_threads=int(cfg.get("num_threads", 3)),
)

_target_frames = mel_time_frames(
    int(cfg.get("sample_rate", 16000)),
    float(cfg.get("audio_duration_sec", 10.0)),
    int(cfg.get("hop_length", 512)),
)
mel_transform = MelTransform(
    sr=int(cfg.get("sample_rate", 16000)),
    n_fft=int(cfg.get("n_fft", 2048)),
    hop_length=int(cfg.get("hop_length", 512)),
    n_mels=int(cfg.get("n_mels", N_MELS)),
    fmin=float(cfg.get("fmin", 20.0)),
    fmax=float(cfg.get("fmax", 8000.0)),
    center=True,
    target_time_frames=_target_frames,
)

app = FastAPI(
    title="MIMII Edge Inference",
    description="ONNX INT8 anomaly detection on Raspberry Pi 5",
    version="1.0.0",
)


class MelPredictRequest(BaseModel):
    """Pre-computed mel spectrogram: flat list or nested [1,1,128,T]."""

    data: List[float] = Field(..., description="Mel values, length = 128 * T (T=313)")
    shape: Optional[List[int]] = Field(
        default=None,
        description="Optional shape override, e.g. [1, 1, 128, 313]",
    )
    run_autoencoder: bool = Field(default=True)


class WaveformPredictRequest(BaseModel):
    """Raw float waveform in [-1, 1], 10 s @ 16 kHz = 160000 samples."""

    samples: List[float]
    run_autoencoder: bool = True


def _mel_to_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis, :, :]
    elif arr.ndim == 3:
        arr = arr[np.newaxis, :, :, :]
    elif arr.ndim != 4:
        raise HTTPException(400, f"Expected mel rank 2–4, got shape {arr.shape}")
    if arr.shape[2] != N_MELS:
        raise HTTPException(400, f"Expected n_mels={N_MELS}, got {arr.shape[2]}")
    if arr.shape[3] != _target_frames:
        raise HTTPException(
            400,
            f"Expected time frames={_target_frames}, got {arr.shape[3]}",
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": cfg.get("classifier_model"),
        "target_frames": _target_frames,
        "stats": engine.get_stats(),
    }


@app.post("/predict")
def predict_mel(req: MelPredictRequest):
    """Run inference on a pre-computed mel spectrogram."""
    shape = req.shape or [1, 1, N_MELS, _target_frames]
    try:
        mel = np.array(req.data, dtype=np.float32).reshape(shape)
    except ValueError as e:
        raise HTTPException(400, f"Invalid mel shape: {e}") from e
    mel = _mel_to_batch(mel)
    return engine.infer(mel, run_autoencoder=req.run_autoencoder)


@app.post("/predict/waveform")
def predict_waveform(req: WaveformPredictRequest):
    """Accept raw audio samples; mel is computed on-device."""
    audio = np.array(req.samples, dtype=np.float32).flatten()
    expected = int(cfg.get("sample_rate", 16000) * cfg.get("audio_duration_sec", 10.0))
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    elif len(audio) > expected:
        audio = audio[:expected]
    mel = mel_transform(audio)
    mel = mel[np.newaxis, :, :, :]
    return engine.infer(mel, run_autoencoder=req.run_autoencoder)


@app.post("/predict/audio")
async def predict_audio_file(
    file: UploadFile = File(...),
    run_autoencoder: bool = True,
):
    """Upload a WAV file (16 kHz mono recommended)."""
    try:
        import soundfile as sf
    except ImportError as e:
        raise HTTPException(
            501,
            "soundfile not installed. pip install soundfile",
        ) from e

    raw = await file.read()
    import io

    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    target_sr = int(cfg.get("sample_rate", 16000))
    if sr != target_sr:
        try:
            import librosa
        except ImportError as e:
            raise HTTPException(501, "librosa required for resampling") from e
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    expected = int(target_sr * float(cfg.get("audio_duration_sec", 10.0)))
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    else:
        audio = audio[:expected]

    mel = mel_transform(audio)
    mel = mel[np.newaxis, :, :, :]
    result = engine.infer(mel, run_autoencoder=run_autoencoder)
    result["source"] = file.filename
    return result
