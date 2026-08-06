# Environment — CHAAD Project

> Last updated: 2026-07-21

## Operating System

| Field | Value | Status |
|-------|-------|--------|
| Primary development OS | Windows 11 | VERIFIED |
| Cross-platform support | Linux (partially) | IMPLEMENTED BUT UNVERIFIED |
| Raspberry Pi | Target for edge deployment | PLANNED |

## Python Environment

| Field | Value | Status |
|-------|-------|--------|
| Python version | 3.11+ | VERIFIED (from `.venv` path) |
| Virtual environment | `.venv/` in project root | VERIFIED |
| Package manager | pip | VERIFIED |

## GPU / CUDA

| Field | Value | Status |
|-------|-------|--------|
| GPU | NVIDIA RTX 4070 SUPER (12.9GB VRAM) | UNVERIFIED (from config comments) |
| CUDA availability | Assumed available | UNVERIFIED (no CUDA check output available) |
| cuDNN | Version UNKNOWN | UNKNOWN |
| Mixed precision | FP16 via torch.cuda.amp, enabled by default | IMPLEMENTED |

## Package Installation

Installation command:

```bash
pip install -r requirements.txt
```

Key dependencies (from `requirements.txt`):
- torch, torchaudio, torchvision (PyTorch 2.x)
- numpy, scipy, pandas
- scikit-learn
- soundfile (WAV loading)
- tqdm (progress bars)
- PyYAML (config parsing)
- tensorboard, wandb (optional logging)

## Dataset Path

| Field | Value | Status |
|-------|-------|--------|
| Configured path | `E:\MIMII` | UNVERIFIED (hardcoded, existence unknown) |
| Override mechanism | CLI argument on `calibrate.py`, environment variable `MIMII_DATASET_DIR` | IMPLEMENTED |
| Manifest path | `metadata/dataset_manifest.csv` | VERIFIED |
| Manifest checksum | `metadata/dataset_manifest.sha256` | VERIFIED |

## Checkpoint Paths

| Field | Value | Status |
|-------|-------|--------|
| Training checkpoints | `checkpoints/epoch_*.pt`, `checkpoints/best_model.pt` | UNVERIFIED (may not exist) |
| Calibration | `checkpoints/detector_calibration.pt` | UNVERIFIED |
| Evaluation report | `checkpoints/eval_report.json` | VERIFIED (legacy, validation-only) |
| Threshold metadata | `artifacts/threshold_metadata.json` | VERIFIED |
| Model manifest | `artifacts/models/manifest.json` | VERIFIED |

## Known Windows Limitations

1. **num_workers must be 0**: PyTorch DataLoader multiprocessing fails on Windows with `num_workers > 0`. Enforced in `config.py`.
2. **Path separators**: Code uses `os.path` for cross-platform compatibility, but `config.py` hardcodes `E:\MIMII` as a raw string.
3. **Triton unavailable**: `torch.compile()` is skipped on Windows because Triton is not available.
4. **Shell syntax**: Commands in documentation use `;` separator (PowerShell-compatible). Linux requires `&&`.

## Verifying Your Environment

```bash
# Check Python version
python --version

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify dataset structure
python verify_dataset.py

# Check manifest integrity
python -c "from utils.split_utils import load_manifest_split; m=load_manifest_split('metadata/dataset_manifest.csv','train'); print(m.counts)"

# Run research integrity audit (no dataset needed)
python _audit_check.py
```

## Reproducible Setup Commands

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify dataset path
# Edit config.py line 14: dataset_dir = r"E:\MIMII"
# Or set environment variable:
# set MIMII_DATASET_DIR=E:\MIMII  (Windows)
# export MIMII_DATASET_DIR=/path/to/MIMII  (Linux)

# 4. Verify dataset
python verify_dataset.py

# 5. Run audit
python _audit_check.py
```

## Common Environment Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: E:\MIMII` | Dataset not at configured path | Update `config.py` or set `MIMII_DATASET_DIR` |
| `RuntimeError: DataLoader worker (pid) exited unexpectedly` | Windows multiprocessing | Ensure `num_workers=0` in config |
| `torch.compile() not supported on Windows` | Triton missing | Expected — training uses eager mode on Windows |
| `soundfile.LibsndfileError` | Missing system library | Install `libsndfile` (Windows: included in soundfile wheel) |

---

*Unknown fields: exact CUDA/cuDNN versions, GPU confirmation, dataset presence. These require environment inspection to resolve.*
