# Project Context — CHAAD

## Identity

**CHAAD** (Configurable Hybrid Acoustic Anomaly Detection) is an AI/ML research project for detecting anomalous sounds from industrial machinery.

## Problem Statement

Industrial machines (fans, pumps, sliders, valves) produce characteristic sounds. When they malfunction, the acoustic signature changes. Manual inspection is expensive, dangerous, and inconsistent. Automated acoustic anomaly detection can reduce downtime and prevent catastrophic failures — but current methods either use fixed fusion rules that don't adapt to varying machine types and noise conditions, or report metrics that may be inflated by data leakage.

## Motivation

- Enable early detection of machine faults from audio alone
- Make anomaly scores **condition-aware** — different signals are reliable under different operating conditions
- Ensure the research is **scientifically rigorous** — no data leakage, reproducible, with strong baselines and ablations
- Produce a publishable conference paper

## Primary Objective

Demonstrate that a **reliability-aware fusion module** — which learns sample-dependent weights for combining multiple anomaly signals conditioned on machine type and noise condition — improves anomaly detection over fixed-weight fusion, equal-weight fusion, and global learned weights, with statistically significant results on a held-out test set.

## Target Users

- ML researchers in industrial anomaly detection
- Factory automation engineers
- Edge/IoT deployment teams

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch 2.x, torchaudio, torchvision |
| Backbone | EfficientNet-B4 (pretrained, adapted to 1-channel mel spectrograms) |
| Temporal | Transformer Encoder (default) or BiLSTM |
| Loss | BCE + SupCon + MSE Reconstruction |
| Audio | Librosa, soundfile, torchaudio transforms |
| Metrics | scikit-learn (ROC-AUC, PR-AUC, pAUC, EER) |
| Calibration | Ledoit-Wolf covariance, z-score normalization |
| Deployment | ONNX export, Docker, Raspberry Pi edge |
| Dataset | MIMII (4 machine types × 4 machine IDs × 3 SNR levels) |

## Repository Structure

```
CHAAD/
├── AGENTS.md              # AI agent rules
├── config.py              # All configuration
├── train.py               # Training entry point
├── calibrate.py            # Detector calibration
├── evaluate.py             # Model evaluation
├── data/dataset.py         # MIMII dataset loader
├── models/
│   ├── hybrid_model.py     # CNN+Transformer+AE architecture
│   └── reliability.py      # Novel reliability-aware fusion
├── inference/detector.py   # Anomaly detection engine
├── training/
│   ├── trainer.py          # Training loop
│   └── loss.py             # Multi-objective loss
├── utils/                  # Audio, metrics, splits, seed, validation
├── scripts/                # Audit, baselines, statistics, subgroups
├── edge_deploy/            # Raspberry Pi deployment
├── docs/                   # Documentation
├── metadata/               # Dataset manifest (source of truth)
├── artifacts/              # Audit reports, calibration metadata
└── reports/                # Evaluation outputs
```

## Scope

**In scope:** Acoustic anomaly detection for industrial machines using supervised deep learning with multi-signal fusion, rigorous evaluation, and edge deployment.

**Out of scope:** Unsupervised methods, non-acoustic sensors, real-time video, cloud-only deployment, multi-language UI.

## Constraints

- Dataset must be MIMII (or compatible structure)
- Windows development environment (num_workers=0 required)
- GPU recommended but CPU fallback supported
- Splits are **machine-independent** and locked via manifest

## Assumptions

- Audio is 16kHz mono, 10-second segments
- Labels are supervised (normal/abnormal per sample)
- Each machine ID represents a physically distinct machine
- Noise conditions are additive environmental noise at known SNR levels

---

*Based on verified source code inspection, executed audit scripts, and Git history.*
