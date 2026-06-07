"""
edge_deploy/edge_streaming.py
────────────────────────────────────────────────────────
Real-time 24/7 streaming inference engine for RPi5.

Architecture:
  - Lock-free SPSC ring buffer (10s @ 16kHz = 160K int16 samples)
  - Audio capture thread via sounddevice (ALSA backend)
  - Inference thread with 5s hop stride (50% overlap)
  - MQTT publish for alerts
  - systemd watchdog integration (sd_notify)
  - Thermal-aware duty cycle management

Usage:
    python edge_streaming.py --config config.yaml

    # As systemd service:
    sudo systemctl start mimii-detector
"""

from __future__ import annotations

import os
import sys
import gc
import time
import json
import signal
import logging
import threading
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "audio_duration_sec": 10.0,
    "hop_duration_sec": 5.0,
    "capture_chunk_samples": 4096,

    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 20.0,
    "fmax": 8000.0,

    "classifier_model": "models/classifier_int8.onnx",
    "autoencoder_model": "models/autoencoder_fp32.onnx",
    "calibration_path": "models/calibration.npz",
    "num_threads": 3,

    "autoencoder_duty_cycle": 4,

    "mqtt_broker": "localhost",
    "mqtt_port": 1883,
    "mqtt_topic": "mimii/machine/status",
    "mqtt_enabled": False,

    "audio_device": None,

    "thermal_warning_c": 75,
    "thermal_critical_c": 82,
    "watchdog_enabled": True,

    "log_level": "INFO",
    "log_file": None,
}


def load_config(path: Optional[str] = None) -> dict:
    """Load config from YAML or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                user_cfg = yaml.safe_load(f) or {}
            config.update(user_cfg)
            print(f"[Stream] Config loaded from {path}")
        except ImportError:
            print("[Stream] PyYAML not installed, using defaults")
    return config


# ─────────────────────────────────────────────────────────
#  Ring Buffer
# ─────────────────────────────────────────────────────────

class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer.

    Stores audio as int16 to halve memory footprint.
    Thread-safe for exactly 1 writer + 1 reader (SPSC pattern).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = np.zeros(capacity, dtype=np.int16)
        self._write_pos = 0
        self._read_pos = 0
        self._total_written = 0

    @property
    def available(self) -> int:
        """Number of samples available for reading."""
        return self._total_written - self._read_pos

    def write(self, data: np.ndarray):
        """Write audio samples (producer thread)."""
        n = len(data)
        if n == 0:
            return

        # Convert float32 [-1,1] to int16 if needed
        if data.dtype == np.float32:
            data = (data * 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)

        pos = self._write_pos % self.capacity
        end = pos + n

        if end <= self.capacity:
            self.buf[pos:end] = data
        else:
            split = self.capacity - pos
            self.buf[pos:] = data[:split]
            self.buf[:n - split] = data[split:]

        self._write_pos += n
        self._total_written += n

    def read(self, n: int) -> Optional[np.ndarray]:
        """Read n samples as float32 [-1,1] (consumer thread)."""
        if self.available < n:
            return None

        pos = self._read_pos % self.capacity
        end = pos + n

        if end <= self.capacity:
            result = self.buf[pos:end].copy()
        else:
            split = self.capacity - pos
            result = np.concatenate([
                self.buf[pos:],
                self.buf[:n - split]
            ])

        self._read_pos += n
        return result.astype(np.float32) / 32767.0

    def peek(self, n: int) -> Optional[np.ndarray]:
        """Peek at n samples without advancing read pointer."""
        if self.available < n:
            return None

        pos = self._read_pos % self.capacity
        end = pos + n

        if end <= self.capacity:
            result = self.buf[pos:end].copy()
        else:
            split = self.capacity - pos
            result = np.concatenate([
                self.buf[pos:],
                self.buf[:n - split]
            ])

        return result.astype(np.float32) / 32767.0

    def advance(self, n: int):
        """Advance read pointer by n samples (after peek + process)."""
        self._read_pos += n


# ─────────────────────────────────────────────────────────
#  Mel Spectrogram (numpy-only, no PyTorch)
# ─────────────────────────────────────────────────────────

def mel_time_frames(
    sample_rate: int,
    audio_duration_sec: float,
    hop_length: int,
) -> int:
    """Time frames for exported ONNX — matches training / edge_quantize."""
    return int(sample_rate * audio_duration_sec / hop_length) + 1


class MelTransform:
    """Pre-computed mel filterbank transform using numpy FFT.

    Avoids librosa import overhead by computing filterbank at init
    and using direct STFT → mel projection.

    Uses center=True padding (same as torchaudio MelSpectrogram) so frame
    count matches exported ONNX (313 for 10 s @ 16 kHz, hop 512).
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float,
        fmax: float,
        center: bool = True,
        target_time_frames: Optional[int] = None,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.target_time_frames = target_time_frames
        self.window = np.hanning(n_fft).astype(np.float32)

        # Build mel filterbank
        self.mel_basis = self._mel_filterbank(sr, n_fft, n_mels, fmin, fmax)

    def _mel_filterbank(self, sr, n_fft, n_mels, fmin, fmax) -> np.ndarray:
        """Compute mel filterbank matrix (n_mels × (n_fft/2+1))."""
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_freqs = n_fft // 2 + 1
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center != left:
                    fbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    fbank[i, j] = (right - j) / (right - center)

        return fbank

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Compute normalized log-mel spectrogram.

        Args:
            audio: (T,) float32 waveform in [-1, 1]

        Returns:
            (1, n_mels, n_frames) float32 in [0, 1]
        """
        if self.center:
            pad = self.n_fft // 2
            audio = np.pad(audio, (pad, pad), mode="reflect")

        # STFT
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, self.n_fft),
            strides=(audio.strides[0] * self.hop_length, audio.strides[0]),
        ).copy()

        # Window + FFT
        frames *= self.window
        spectrum = np.fft.rfft(frames, n=self.n_fft)
        power = np.abs(spectrum) ** 2

        # Mel projection
        mel = self.mel_basis @ power.T  # (n_mels, n_frames)

        # Log + normalize to [0, 1] (matching training pipeline)
        mel = np.maximum(mel, 1e-10)
        log_mel = 10.0 * np.log10(mel)
        top_db = 80.0
        log_mel = np.maximum(log_mel, log_mel.max() - top_db)
        normalized = (log_mel - log_mel.max() + top_db) / top_db
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = normalized[np.newaxis, :, :]

        if self.target_time_frames is not None:
            t = self.target_time_frames
            cur = normalized.shape[2]
            if cur < t:
                pad = np.zeros((1, self.n_mels, t - cur), dtype=np.float32)
                normalized = np.concatenate([normalized, pad], axis=2)
            elif cur > t:
                normalized = normalized[:, :, :t]

        return normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────
#  Thermal Monitor
# ─────────────────────────────────────────────────────────

def read_cpu_temp() -> float:
    """Read CPU temperature on RPi5 (returns -1 if unavailable)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return -1.0


# ─────────────────────────────────────────────────────────
#  MQTT Publisher
# ─────────────────────────────────────────────────────────

class MQTTPublisher:
    """Lightweight MQTT client for alert publishing."""

    def __init__(self, broker: str, port: int, topic: str):
        self.topic = topic
        self._client = None
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="mimii-detector",
            )
            self._client.connect(broker, port, keepalive=60)
            self._client.loop_start()
            print(f"[MQTT] Connected to {broker}:{port}")
        except ImportError:
            print("[MQTT] paho-mqtt not installed, publishing disabled")
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")

    def publish(self, result: Dict[str, Any]):
        if self._client is None:
            return
        try:
            payload = json.dumps(result, default=str)
            self._client.publish(self.topic, payload, qos=1)
        except Exception as e:
            logging.warning(f"MQTT publish error: {e}")

    def close(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()


# ─────────────────────────────────────────────────────────
#  Streaming Engine
# ─────────────────────────────────────────────────────────

class StreamingEngine:
    """Main 24/7 streaming inference engine."""

    def __init__(self, config: dict):
        self.cfg = config
        self._running = False
        self._logger = self._setup_logging()

        # Audio params
        sr = config["sample_rate"]
        duration = config["audio_duration_sec"]
        hop_dur = config["hop_duration_sec"]
        self.window_samples = int(sr * duration)
        self.hop_samples = int(sr * hop_dur)

        # Ring buffer — 2x window for safety margin
        self.ring = RingBuffer(self.window_samples * 2)

        # Mel transform
        target_frames = mel_time_frames(sr, duration, config["hop_length"])
        self.mel_transform = MelTransform(
            sr=sr,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            center=True,
            target_time_frames=target_frames,
        )

        # Inference engine (lazy init)
        self.engine = None

        # MQTT
        self.mqtt: Optional[MQTTPublisher] = None
        if config.get("mqtt_enabled"):
            self.mqtt = MQTTPublisher(
                config["mqtt_broker"],
                config["mqtt_port"],
                config["mqtt_topic"],
            )

        # Watchdog
        self._notifier = None
        if config.get("watchdog_enabled"):
            try:
                import sdnotify
                self._notifier = sdnotify.SystemdNotifier()
            except ImportError:
                pass

        # Stats
        self._inference_count = 0
        self._ae_counter = 0

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("mimii-stream")
        logger.setLevel(getattr(logging, self.cfg.get("log_level", "INFO")))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)

        log_file = self.cfg.get("log_file")
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(handler.formatter)
            logger.addHandler(fh)

        return logger

    def _init_engine(self):
        """Initialize ONNX Runtime engine (deferred to avoid import at module level)."""
        # Ensure edge_deploy dir is on path so edge_runtime can be imported
        # regardless of the working directory when the service starts
        _edge_dir = str(Path(__file__).resolve().parent)
        if _edge_dir not in sys.path:
            sys.path.insert(0, _edge_dir)

        from edge_runtime import EdgeInferenceEngine

        self.engine = EdgeInferenceEngine(
            classifier_model=self.cfg["classifier_model"],
            autoencoder_model=self.cfg.get("autoencoder_model"),
            calibration_path=self.cfg.get("calibration_path"),
            num_threads=self.cfg.get("num_threads", 3),
        )
        self._logger.info("Inference engine initialized")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio capture thread."""
        if status:
            self._logger.warning(f"Audio status: {status}")
        # indata: (frames, channels) float32 or int16
        audio = indata[:, 0] if indata.ndim > 1 else indata
        self.ring.write(audio.flatten())

    def _check_thermal(self) -> str:
        """Check CPU temp and return throttle level."""
        temp = read_cpu_temp()
        if temp < 0:
            return "ok"
        if temp >= self.cfg["thermal_critical_c"]:
            self._logger.error(f"THERMAL CRITICAL: {temp:.1f}°C — skipping inference")
            return "critical"
        if temp >= self.cfg["thermal_warning_c"]:
            self._logger.warning(f"THERMAL WARNING: {temp:.1f}°C")
            return "warning"
        return "ok"

    def _process_window(self):
        """Process one audio window through the inference pipeline."""
        audio = self.ring.peek(self.window_samples)
        if audio is None:
            return

        self.ring.advance(self.hop_samples)

        # Thermal check
        thermal = self._check_thermal()
        if thermal == "critical":
            time.sleep(5)
            return

        # Mel transform
        mel = self.mel_transform(audio)
        mel = mel[np.newaxis, :, :, :]  # (1, 1, n_mels, T)

        # Decide whether to run autoencoder this cycle
        self._ae_counter += 1
        run_ae = (self._ae_counter % self.cfg.get("autoencoder_duty_cycle", 4)) == 0

        # Inference
        result = self.engine.infer(mel, run_autoencoder=run_ae)
        self._inference_count += 1

        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result["cpu_temp_c"] = round(read_cpu_temp(), 1)
        result["cycle"] = self._inference_count

        # Log
        score = result["anomaly_score"]
        label = result["label"]
        lat = result["inference_ms"]
        self._logger.info(
            f"[{self._inference_count}] {label} "
            f"score={score:.4f} latency={lat:.0f}ms "
            f"temp={result['cpu_temp_c']}°C"
        )

        # Alert on anomaly
        if result["label"] == "Anomalous":
            self._logger.warning(
                f"ANOMALY DETECTED: score={score:.4f} "
                f"severity={result['severity']} "
                f"fault={result['fault_type']}"
            )

        # Publish
        if self.mqtt:
            self.mqtt.publish(result)

        # Watchdog heartbeat
        if self._notifier:
            self._notifier.notify("WATCHDOG=1")
            self._notifier.notify(f"STATUS=score={score:.4f} health={result['system']['health_index']}")

        # Periodic GC
        if self._inference_count % 1000 == 0:
            gc.collect()
            self._logger.info(f"GC complete. Stats: {self.engine.get_stats()}")

    def run(self):
        """Main loop — blocks forever (or until SIGTERM)."""
        self._running = True

        # Signal handlers
        def _shutdown(signum, frame):
            self._logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # Init engine
        self._init_engine()

        # Notify systemd we're ready
        if self._notifier:
            self._notifier.notify("READY=1")

        # Start audio capture
        import sounddevice as sd

        stream_kwargs = {
            "samplerate": self.cfg["sample_rate"],
            "channels": 1,
            "dtype": "float32",
            "blocksize": self.cfg["capture_chunk_samples"],
            "callback": self._audio_callback,
        }
        if self.cfg.get("audio_device") is not None:
            stream_kwargs["device"] = self.cfg["audio_device"]

        self._logger.info(
            f"Starting audio capture: {self.cfg['sample_rate']}Hz, "
            f"chunk={self.cfg['capture_chunk_samples']}"
        )

        try:
            with sd.InputStream(**stream_kwargs):
                self._logger.info("Audio stream active. Inference loop starting...")

                # Wait for initial buffer fill
                while self._running and self.ring.available < self.window_samples:
                    time.sleep(0.1)

                self._logger.info("Buffer filled. Running inference...")

                while self._running:
                    if self.ring.available >= self.window_samples:
                        self._process_window()
                    else:
                        time.sleep(0.05)

        except Exception as e:
            self._logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self._logger.info("Shutting down...")
            if self.mqtt:
                self.mqtt.close()
            if self._notifier:
                self._notifier.notify("STOPPING=1")
            self._logger.info(f"Total inferences: {self._inference_count}")
            if self.engine:
                stats = self.engine.get_stats()
                self._logger.info(f"Performance: {json.dumps(stats, indent=2)}")


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIMII Real-Time Streaming Inference")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--model", help="Override classifier model path")
    parser.add_argument("--device", help="Audio device name or index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    config = load_config(args.config)
    if args.model:
        config["classifier_model"] = args.model
    if args.device:
        try:
            config["audio_device"] = int(args.device)
        except ValueError:
            config["audio_device"] = args.device

    engine = StreamingEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
