# Raspberry Pi 5 — Production Deployment Guide

This document is the **authoritative** edge deployment path for this repo.  
Quantization runs on your **GPU PC**; the Pi only runs **ONNX Runtime INT8**.

## Architecture

```
GPU PC (train / calibrate / quantize)
  ├── checkpoints/best_model.pt
  ├── checkpoints/detector_calibration.pt
  └── python edge_deploy/setup_edge.py
           │
           ▼
  edge_deploy/models/
    ├── classifier_int8.onnx   ← primary runtime
    ├── classifier_fp16.onnx
    ├── classifier_fp32.onnx
    ├── autoencoder_fp32.onnx
    └── calibration.npz
           │
           │  scp / WinSCP
           ▼
Raspberry Pi 5 (/opt/mimii)
  ├── ONNX Runtime (ARM + XNNPACK)
  ├── edge_streaming.py      ← 24/7 mic (systemd)
  ├── inference_server.py    ← optional HTTP API
  └── edge_watchdog.py       ← thermal / disk health
```

## Hardware checklist

| Item | Recommendation |
|------|----------------|
| Board | Raspberry Pi 5, **8 GB** RAM |
| Storage | 128 GB SSD (preferred) or U3/A2 microSD |
| Cooling | Active fan + heatsink (INT8 inference is thermally heavy) |
| PSU | Official **27 W** USB-C |

## Phase 1 — Export on GPU PC (do not quantize on Pi)

```powershell
cd E:\ASP
python edge_deploy/setup_edge.py
```

Requires `checkpoints/best_model.pt` and `checkpoints/detector_calibration.pt`.  
Produces ONNX artifacts under `edge_deploy/models/`.

## Phase 2 — Copy to Pi

From Windows (replace `PI_IP`):

```bash
ssh pi@PI_IP "sudo mkdir -p /opt/mimii/models /var/log/mimii && sudo chown -R pi:pi /opt/mimii /var/log/mimii"

scp -r edge_deploy/models/* pi@PI_IP:/opt/mimii/models/
scp edge_deploy/*.py edge_deploy/config.yaml edge_deploy/requirements_rpi.txt edge_deploy/deploy.sh pi@PI_IP:/opt/mimii/
scp -r edge_deploy/systemd pi@PI_IP:/opt/mimii/
```

Or run `python edge_deploy/setup_edge.py` and copy the printed `scp` commands.

## Phase 3 — Pi OS setup

1. Install **Raspberry Pi OS 64-bit Lite** (no desktop).
2. On the Pi:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git cmake build-essential \
  libatlas-base-dev libopenblas-dev libffi-dev libssl-dev htop curl wget \
  cpufrequtils zram-tools

cd /opt/mimii
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_rpi.txt
chmod +x deploy.sh

# Performance (optional but recommended)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo "gpu_mem=64" | sudo tee -a /boot/firmware/config.txt
```

## Phase 4 — Validate before production

```bash
cd /opt/mimii && source venv/bin/activate

# Latency (target INT8: ~80–250 ms on Pi 5)
python3 edge_benchmark.py --model models/classifier_int8.onnx --warmup 20 --iterations 200

# Mel shape must be 128×313
python3 edge_benchmark.py --mel

# FP32 vs INT8 drift check
python3 edge_benchmark.py --validate \
  --fp32 models/classifier_fp32.onnx \
  --int8 models/classifier_int8.onnx
```

## Phase 5 — Run services

### 24/7 streaming (factory floor)

```bash
python3 edge_streaming.py --list-devices
python3 edge_streaming.py --config config.yaml   # test run

sudo cp /opt/mimii/systemd/mimii-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mimii-detector
journalctl -fu mimii-detector
```

### HTTP API (integration / remote clients)

```bash
sudo cp /opt/mimii/systemd/mimii-api.service /etc/systemd/system/
sudo systemctl enable --now mimii-api
curl http://localhost:8000/health
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status + latency stats |
| POST | `/predict` | Pre-computed mel `(1,1,128,313)` |
| POST | `/predict/waveform` | Raw float samples |
| POST | `/predict/audio` | WAV upload |

### OTA model updates

Copy a new `classifier_int8.onnx` into `models/`. Either restart manually or:

```bash
nohup python3 auto_update.py &
```

## Phase 6 — Updates from git

```bash
cd /opt/mimii && ./deploy.sh
```

## Expected performance (Pi 5)

| Format | Typical latency |
|--------|-----------------|
| FP32 | 400–900 ms |
| FP16 | 250–500 ms |
| **INT8** | **80–250 ms** |

Thermal throttling increases latency; run `edge_benchmark.py --soak-hours 4` before go-live.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| ONNX shape error | Mel must be **313** time frames; fixed in `MelTransform` (center padding) |
| High latency | `export OMP_NUM_THREADS=3`, performance governor, active cooling |
| No audio | `edge_streaming.py --list-devices`, set `audio_device` in `config.yaml` |
| Service crash | `journalctl -u mimii-detector -e` |

## Future upgrades

- **Hailo-8** or **Google Coral** USB accelerator for sub-50 ms latency
- Model pruning / distillation for lighter EfficientNet + transformer stack
