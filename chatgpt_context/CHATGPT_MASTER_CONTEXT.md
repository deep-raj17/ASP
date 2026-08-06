# CHAAD Project — Master Context for ChatGPT

> Paste this into a ChatGPT conversation or Project for instant project understanding. Full documentation at `docs/`.

## Project Identity

**CHAAD** (Configurable Hybrid Acoustic Anomaly Detection) — AI/ML research project for detecting anomalous sounds from industrial machines (fans, pumps, sliders, valves) using the MIMII dataset. Target: conference publication.

## Core Problem

Industrial machines produce characteristic sounds. When they malfunction, the acoustic signature changes. Current anomaly detection systems use **fixed fusion rules** that don't adapt to different machine types or noise conditions.

## Novel Contribution

A **reliability-aware fusion module** (`models/reliability.py`) that learns sample-dependent weights for combining four anomaly signals (reconstruction, embedding distance, Mahalanobis distance, contrastive distance), conditioned on machine type and noise condition. Differs from fixed-weight fusion, equal-weight fusion, and global learned weights.

## Tech Stack

Python 3.11+ | PyTorch 2.x | EfficientNet-B4 | Transformer Encoder | Autoencoder | MIMII dataset (53,046 WAV files) | scikit-learn | ONNX/Docker/Raspberry Pi deployment

## Architecture (Simplified)

```
Audio (16kHz, 10s) → Mel Spectrogram → CNN (EfficientNet-B4) + Autoencoder
    → Transformer → Attention Pool → [Classifier + Projector]
    → 4 Anomaly Signals → Reliability Gate → Fused Score → Threshold → Anomaly/Not
```

## Dataset Protocol (CRITICAL)

- **Machine-independent 3-split**: Each machine ID in exactly one split
- train: id_04 (12,045 samples) | val: id_00+id_02 (28,254) | test: id_06 (12,747)
- **Manifest** (`metadata/dataset_manifest.csv`) is the single source of truth
- Threshold selected on **validation only** (Youden's J), frozen for test
- Calibration on **train_normal only**

## Verified State (2026-07-21)

| Check | Status |
|-------|--------|
| Data leakage (machine, SHA-256, segments) | ✅ PASS |
| Normalization/Calibration leakage | ✅ PASS |
| Threshold protocol (3-split) | ✅ PASS |
| Metric implementation (continuous scores) | ✅ PASS |
| Shortcut learning (metadata AUC=0.59) | ✅ PASS |
| Reproducibility (seeds, provenance) | ✅ IMPLEMENTED |
| Model checkpoint available | ❌ BLOCKED (needs dataset at `E:\MIMII`) |
| Test-set evaluation | ❌ BLOCKED (needs checkpoint) |
| Baseline comparisons (11 baselines) | ❌ BLOCKED (needs checkpoint) |
| Publication audit score | 57.1% (CONDITIONAL) |

## Active Blockers

1. **Dataset**: Unknown if MIMII is at `E:\MIMII`
2. **Checkpoint**: No `checkpoints/best_model.pt` exists

## Important Paths

| Path | Purpose |
|------|---------|
| `config.py` | All configuration (edit dataset_dir first) |
| `train.py` | Training entry point |
| `evaluate.py` | Model evaluation (supports --split test) |
| `models/reliability.py` | Novel contribution |
| `metadata/dataset_manifest.csv` | Split truth |
| `_audit_check.py` | Data integrity audit |
| `scripts/run_publication_audit.py` | Go/no-go framework |

## Rules for Answering

1. Distinguish VERIFIED from UNVERIFIED claims
2. Never cite legacy metrics (ROC-AUC 99.99997%) as final — they're from an unknown split protocol
3. Model checkpoint and dataset availability are UNKNOWN — do not assume they exist
4. Source code > documentation for truth
5. All performance metrics must come from test split (id_06) with frozen validation threshold

## Next Action

Train model (`python train.py`) then run baselines, statistics, ablations, and robustness analysis to reach GO verdict (>90% publication audit score).

---

*Full docs: `AGENTS.md`, `docs/AI_CONTEXT_INDEX.md`, `docs/CURRENT_STATE.md`*
