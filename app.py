"""Gradio UI for the MIMII acoustic anomaly detector.

The application is inference-only:
- loads existing trained weights from checkpoints/artifacts
- loads existing calibration from checkpoints
- does not train, calibrate, evaluate, or modify model artifacts

Startup stays fast by loading the neural network only when the first audio file
is analysed.
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("APP_USE_TORCH_COMPILE", "0")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

STARTED_AT = time.perf_counter()
ROOT = Path(__file__).resolve().parent

print("[App] Starting MIMII Acoustic Anomaly Detector...", flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gradio as gr

from config import cfg
from paths import artifacts_status, calibration_path, resolve_model_checkpoint

print(f"[App] UI imports ready in {time.perf_counter() - STARTED_AT:.2f}s", flush=True)


CKPT_PATH, CKPT_PRECISION = resolve_model_checkpoint(prefer_fp16_on_cuda=True)
CALIB_PATH = calibration_path()
TARGET_LEN = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)


@dataclass
class RuntimeState:
    device: Any | None = None
    detector: Any | None = None
    processor: Any | None = None
    ready: bool = False


STATE = RuntimeState()


def _find_available_port(start_port: int = 7860, max_tries: int = 20) -> int:
    """Find a free localhost TCP port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            exclusive_addr_use = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive_addr_use is not None:
                sock.setsockopt(socket.SOL_SOCKET, exclusive_addr_use, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found from {start_port} to {start_port + max_tries - 1}")


def _lazy_init() -> None:
    """Load model, detector, calibration, and audio processor on first use."""
    if STATE.ready:
        return

    import torch

    from inference.production_detector import ProductionDetector
    from models.hybrid_model import HybridAnomalyModel
    from utils.audio_utils import AudioProcessor
    from utils.checkpoint import load_model_weights
    from utils.gpu_utils import get_memory_stats, setup_cuda_optimizations
    from utils.monitoring import SystemMonitor
    from utils.validation import setup_logging

    STATE.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[App] Device: {STATE.device}", flush=True)

    for name, value in artifacts_status().items():
        print(f"[App] artifact {name}: {value}", flush=True)

    model = HybridAnomalyModel(cfg.model).to(STATE.device)
    if os.path.isfile(CKPT_PATH):
        print(f"[App] Loading checkpoint ({CKPT_PRECISION}): {CKPT_PATH}", flush=True)
        epoch, loaded_precision = load_model_weights(model, CKPT_PATH, STATE.device)
        print(
            f"[App] Checkpoint loaded (epoch={epoch}, storage={loaded_precision})",
            flush=True,
        )
    else:
        print(f"[App] WARNING: checkpoint not found: {CKPT_PATH}", flush=True)
        print("[App] Running with randomly initialized weights.", flush=True)

    model.eval()

    use_compile = os.environ.get("APP_USE_TORCH_COMPILE", "0") == "1"
    STATE.detector = ProductionDetector(model, cfg, use_compiled=use_compile)

    if os.path.isfile(CALIB_PATH):
        print(f"[App] Loading calibration: {CALIB_PATH}", flush=True)
        STATE.detector.load_calibration(CALIB_PATH)
        print("[App] Calibration loaded", flush=True)
    else:
        print(f"[App] WARNING: calibration not found: {CALIB_PATH}", flush=True)

    STATE.processor = AudioProcessor(cfg.data)
    setup_cuda_optimizations()
    setup_logging(log_file=str(ROOT / "logs" / "app.log"))

    if STATE.device.type == "cuda":
        memory = get_memory_stats()
        print(f"[App] GPU memory free: {memory.get('free_gb', 0):.2f} GB", flush=True)

    SystemMonitor(interval_seconds=60).start()
    STATE.ready = True
    print("[App] Model initialization complete.", flush=True)


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    buffer.seek(0)
    image = Image.open(buffer).copy()
    buffer.close()
    plt.close(fig)
    return image


def plot_waveform(wav: np.ndarray, sr: int) -> Image.Image:
    times = np.linspace(0, len(wav) / sr, len(wav))
    fig, ax = plt.subplots(figsize=(11, 2.6), facecolor="#101319")
    ax.set_facecolor("#101319")
    ax.plot(times, wav, color="#36c5f0", linewidth=0.55)
    ax.fill_between(times, wav, color="#36c5f0", alpha=0.14)
    ax.set_title("Waveform", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Amplitude", color="#cbd5e1", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return _fig_to_pil(fig)


def plot_spectrogram(mel: np.ndarray, sr: int, hop: int) -> Image.Image:
    frames = mel.shape[1]
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor="#101319")
    ax.set_facecolor("#101319")
    image = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", interpolation="bilinear")
    ticks = np.linspace(0, frames - 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{v:.1f}s" for v in np.linspace(0, frames * hop / sr, 6)],
        color="#cbd5e1",
        fontsize=8,
    )
    ax.set_yticks(np.linspace(0, mel.shape[0] - 1, 5))
    ax.set_yticklabels(
        [f"{int(v)}" for v in np.linspace(cfg.data.fmin, cfg.data.fmax, 5)],
        color="#cbd5e1",
        fontsize=8,
    )
    ax.set_title("Log-Mel Spectrogram", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="#cbd5e1", fontsize=9)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.ax.yaxis.set_tick_params(color="#cbd5e1", labelcolor="#cbd5e1")
    fig.tight_layout()
    return _fig_to_pil(fig)


def plot_timeline(segments: list[dict[str, Any]], duration: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=(11, 1.8), facecolor="#101319")
    ax.set_facecolor("#101319")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.axhspan(0, 1, color="#0f3b2e", alpha=0.7)

    for segment in segments:
        severity = float(segment.get("severity", 0))
        color = "#ef4444" if severity > 0.5 else "#f59e0b"
        ax.axvspan(segment["start"], segment["end"], color=color, alpha=0.78)
        center = (segment["start"] + segment["end"]) / 2
        ax.text(center, 0.5, f"{severity:.2f}", color="#fff", ha="center", va="center", fontsize=8)

    ax.legend(
        handles=[
            mpatches.Patch(color="#0f3b2e", label="Normal"),
            mpatches.Patch(color="#f59e0b", label="Warning"),
            mpatches.Patch(color="#ef4444", label="Critical"),
        ],
        loc="upper right",
        framealpha=0.25,
        labelcolor="#f8fafc",
        fontsize=8,
    )
    ax.set_title("Temporal Anomaly Timeline", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9)
    ax.set_yticks([])
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return _fig_to_pil(fig)


def render_gauge(health: int, risk: str) -> str:
    palette = {
        "Low": ("#10b981", "#ecfdf5", "#064e3b"),
        "Medium": ("#f59e0b", "#fffbeb", "#78350f"),
        "High": ("#f97316", "#fff7ed", "#7c2d12"),
        "Critical": ("#ef4444", "#fef2f2", "#7f1d1d"),
    }
    foreground, background, text = palette.get(risk, palette["Low"])
    width = max(0, min(int(health), 100))
    return f"""
<div class="health-card" style="border-color:{foreground}; background:{background}; color:{text};">
  <div class="health-topline">
    <span>System Health</span>
    <strong style="color:{foreground};">{risk} Risk</strong>
  </div>
  <div class="health-value" style="color:{foreground};">{width}<span>/100</span></div>
  <div class="health-bar"><div style="width:{width}%; background:{foreground};"></div></div>
  <div class="health-label">Health index derived from calibrated acoustic anomaly fusion.</div>
</div>
"""


def generate_output(detector_result: dict[str, Any], native_sr: int, audio_duration: float) -> dict[str, Any]:
    """Create the diagnostic report without changing detector scores."""
    anomaly_score = float(detector_result["anomaly_score"])
    confidence = float(detector_result["confidence"])
    is_anomalous = detector_result["label"] == "Anomalous"

    if anomaly_score > 0.8:
        severity = "high"
    elif anomaly_score > 0.5:
        severity = "medium"
    else:
        severity = "low"

    multi_scores = detector_result.get("multi_scores", {})
    reconstruction = float(multi_scores.get("reconstruction_error", 0))
    mahalanobis = float(multi_scores.get("mahalanobis", 0))

    if anomaly_score > 0.7:
        fault_type = "bearing_fault" if reconstruction > mahalanobis else "mechanical_looseness"
    elif anomaly_score > 0.5:
        fault_type = "early_degradation"
    else:
        fault_type = "normal_variation"

    recommendation = {
        "high": "Immediate inspection required. Stop equipment if possible.",
        "medium": "Schedule maintenance soon and monitor closely.",
        "low": "System operating normally. Continue routine monitoring.",
    }[severity]

    return {
        "machine": "fan",
        "status": "abnormal" if is_anomalous else "normal",
        "confidence": round(confidence, 3),
        "severity": severity,
        "fault_type": fault_type,
        "anomaly_score": round(anomaly_score, 3),
        "recommendation": recommendation,
        "multi_scores": multi_scores,
        "temporal_anomaly": detector_result.get("temporal_anomaly", []),
        "system": detector_result.get("system", {}),
        "audio_info": {
            "native_sample_rate_hz": native_sr,
            "processing_sample_rate_hz": cfg.data.sample_rate,
            "duration_sec": round(audio_duration, 2),
            "resampled": native_sr != cfg.data.sample_rate,
        },
        "model_info": {
            "backbone": cfg.model.backbone,
            "temporal_module": cfg.model.temporal_module,
            "device": str(STATE.device),
            "n_mels": cfg.data.n_mels,
            "n_mfcc": cfg.data.n_mfcc,
            "fmin_hz": cfg.data.fmin,
            "fmax_hz": cfg.data.fmax,
        },
    }


def analyse_audio(audio_path: str | None):
    if not audio_path:
        return None, "<p class='muted'>Upload a WAV file to begin.</p>", None, None, None

    _lazy_init()

    import torch
    import torchaudio

    from utils.audio_utils import pad_or_trim
    from utils.monitoring import record_metric, timed
    from utils.validation import logger, validate_audio_file

    validation = validate_audio_file(audio_path)
    if not validation.valid:
        return None, f"<p class='error'>Error: {validation.error_message}</p>", None, None, None

    with timed("analyse_audio"):
        try:
            waveform, native_sr = torchaudio.load(audio_path)
        except Exception as exc:
            logger.error("Failed to load audio: %s", exc)
            return None, f"<p class='error'>Failed to load audio file: {exc}</p>", None, None, None

        audio_duration = waveform.shape[-1] / native_sr
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if native_sr != cfg.data.sample_rate:
            print(f"[App] Resampling {native_sr} Hz to {cfg.data.sample_rate} Hz", flush=True)
            waveform = torchaudio.functional.resample(waveform, native_sr, cfg.data.sample_rate)

        waveform = pad_or_trim(waveform, TARGET_LEN)
        mel, _mfcc, processed_waveform = STATE.processor(waveform, augment=False)
        mel_batch = mel.unsqueeze(0).to(STATE.device)

        with torch.inference_mode():
            raw_result = STATE.detector.detect_batch(mel_batch)[0]

    report = generate_output(raw_result, native_sr, audio_duration)
    system = raw_result.get("system", {})
    record_metric("inference.anomaly_score", report["anomaly_score"])
    record_metric("inference.health_index", system.get("health_index", 0))

    waveform_image = plot_waveform(processed_waveform[0].numpy(), cfg.data.sample_rate)
    spectrogram_image = plot_spectrogram(mel[0].numpy(), cfg.data.sample_rate, cfg.data.hop_length)
    timeline_image = plot_timeline(raw_result.get("temporal_anomaly", []), cfg.data.audio_duration_sec)
    gauge_html = render_gauge(system.get("health_index", 0), system.get("risk_level", "Low"))

    return report, gauge_html, waveform_image, spectrogram_image, timeline_image


CSS = """
body {
  font-family: Inter, Segoe UI, Arial, sans-serif;
  background: #eef3f8;
  color: #172033;
}
footer { display: none !important; }
.gradio-container {
  max-width: 1380px !important;
  padding: 18px 20px 28px !important;
}
.app-header {
  background: linear-gradient(135deg, #0f172a 0%, #164e63 58%, #0f766e 100%);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 8px;
  color: #f8fafc;
  padding: 24px 26px;
  margin-bottom: 14px;
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.18);
}
.app-header h1 {
  margin: 0 0 8px;
  font-size: 32px;
  line-height: 1.15;
  letter-spacing: 0;
}
.app-header p {
  margin: 0;
  color: #dbeafe;
  font-size: 15px;
  max-width: 860px;
}
.status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 0 0 16px;
}
.status-item {
  background: #ffffff;
  border: 1px solid #d9e3ef;
  border-radius: 8px;
  padding: 12px 14px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}
.status-item span {
  display: block;
  color: #64748b;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.status-item strong {
  display: block;
  margin-top: 4px;
  color: #0f172a;
  font-size: 14px;
}
.section-title {
  margin: 0 0 8px;
  color: #0f172a;
  font-size: 14px;
  font-weight: 800;
}
.panel-note {
  margin: -2px 0 12px;
  color: #64748b;
  font-size: 13px;
}
.control-panel,
.visual-panel {
  background: #ffffff;
  border: 1px solid #d9e3ef;
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}
.control-panel .gr-button {
  min-height: 46px;
  font-weight: 800;
}
.health-card {
  border: 1px solid;
  border-radius: 8px;
  padding: 18px;
  text-align: left;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
}
.health-topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  font-size: 13px;
  font-weight: 700;
}
.health-value {
  font-size: 56px;
  font-weight: 900;
  line-height: 1;
  margin: 14px 0 12px;
  letter-spacing: 0;
}
.health-value span {
  font-size: 22px;
  color: #64748b;
}
.health-label {
  color: #475569;
  margin-top: 10px;
  font-size: 12px;
}
.health-bar {
  height: 12px;
  background: rgba(15, 23, 42, 0.12);
  border-radius: 999px;
  overflow: hidden;
}
.health-bar div {
  height: 100%;
  transition: width 0.35s ease;
}
.muted {
  color: #64748b;
  margin: 0;
  padding: 12px;
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
}
.error {
  color: #b91c1c;
  font-weight: 700;
  margin: 0;
  padding: 12px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
}
.model-strip {
  margin-top: 14px;
  padding: 12px 14px;
  border: 1px solid #d9e3ef;
  border-radius: 8px;
  background: #ffffff;
  color: #334155;
  font-size: 13px;
}
.model-strip p { margin: 0; }
.gradio-container img {
  border-radius: 8px;
}
@media (max-width: 900px) {
  .status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .app-header h1 { font-size: 26px; }
}
@media (max-width: 560px) {
  .status-strip { grid-template-columns: 1fr; }
  .gradio-container { padding: 12px !important; }
}
"""

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MIMII Acoustic Anomaly Detector") as demo:
        gr.HTML(
            """
            <div class="app-header">
              <h1>Industrial Acoustic Anomaly Detection</h1>
              <p>Inference-only diagnostic console for trained MIMII predictive maintenance models. Upload a machine audio sample to inspect health, anomaly evidence, temporal regions, and calibrated model metadata.</p>
            </div>
            """
        )

        gr.HTML(
            f"""
            <div class="status-strip">
              <div class="status-item"><span>Mode</span><strong>Inference only</strong></div>
              <div class="status-item"><span>Weights</span><strong>{CKPT_PRECISION.upper()} artifact</strong></div>
              <div class="status-item"><span>Calibration</span><strong>{'Available' if os.path.isfile(CALIB_PATH) else 'Missing'}</strong></div>
              <div class="status-item"><span>Fusion</span><strong>Recon + Embedding + Mahalanobis + Contrastive</strong></div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=350, elem_classes=["control-panel"]):
                gr.HTML(
                    """
                    <p class="section-title">Analysis Input</p>
                    <p class="panel-note">Upload a WAV file. Existing trained weights and calibration are used; no retraining runs here.</p>
                    """
                )
                audio_input = gr.Audio(type="filepath", label="Upload Audio (.wav)")
                analyse_button = gr.Button("Analyse", variant="primary", size="lg")
                gr.HTML('<p class="section-title">System Health</p>')
                gauge_output = gr.HTML(label="System Health")
                gr.HTML('<p class="section-title">Diagnostic Report</p>')
                json_output = gr.JSON(label="Diagnostic Report")

            with gr.Column(scale=2, elem_classes=["visual-panel"]):
                gr.HTML(
                    """
                    <p class="section-title">Acoustic Evidence</p>
                    <p class="panel-note">Waveform, log-mel spectrogram, and temporal anomaly map are all preserved from the original UI.</p>
                    """
                )
                waveform_output = gr.Image(label="Waveform", type="pil")
                spectrogram_output = gr.Image(label="Mel Spectrogram", type="pil")
                timeline_output = gr.Image(label="Temporal Anomaly Map", type="pil")

        analyse_button.click(
            fn=analyse_audio,
            inputs=[audio_input],
            outputs=[
                json_output,
                gauge_output,
                waveform_output,
                spectrogram_output,
                timeline_output,
            ],
        )

        gr.HTML(
            """
            <div class="model-strip">
              <p><strong>Backbone</strong>: EfficientNet-B4 | <strong>Temporal</strong>: Transformer Encoder | <strong>Scores</strong>: Reconstruction, Mahalanobis, Embedding, Contrastive | <strong>Dataset</strong>: MIMII</p>
            </div>
            """
        )

    return demo


app = build_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MIMII Gradio UI.")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE") == "1")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser on startup.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    port = _find_available_port(args.port)
    if port != args.port:
        print(f"[App] Port {args.port} is busy. Using port {port}.", flush=True)

    local_url = f"http://127.0.0.1:{port}"
    print(f"[App] UI ready in {time.perf_counter() - STARTED_AT:.2f}s", flush=True)
    print(f"[App] Launching at {local_url}", flush=True)
    print("[App] Model loads on first analysis request.", flush=True)

    app.launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        theme=THEME,
        css=CSS,
        inbrowser=not args.no_browser,
        prevent_thread_lock=False,
    )
