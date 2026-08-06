# Reproducibility — CHAAD Project

> Last updated: 2026-07-21

## Seed Control

All random sources are controlled via `utils/seed.py`:

| Source | Controlled? | Method |
|--------|-------------|--------|
| Python `random` | ✅ | `random.seed(seed)` |
| NumPy | ✅ | `np.random.seed(seed)` |
| PyTorch CPU | ✅ | `torch.manual_seed(seed)` |
| PyTorch CUDA | ✅ | `torch.cuda.manual_seed_all(seed)` |
| cuDNN | ✅ | `torch.backends.cudnn.deterministic = True` |
| PYTHONHASHSEED | ✅ | `os.environ["PYTHONHASHSEED"] = str(seed)` |
| DataLoader workers | ✅ | `seed_worker()` via `worker_init_fn` |

Default seed: `42` (configurable in `TrainingConfig.random_seed`).

## Deterministic Settings

| Setting | Value | Impact |
|---------|-------|--------|
| `deterministic_cudnn` | `True` | cuDNN uses deterministic algorithms (slower but reproducible) |
| `cudnn.benchmark` | `False` | Disables adaptive algorithm selection |
| `mixed_precision` | `True` | FP16 via GradScaler, non-deterministic in edge cases |
| DataLoader `shuffle` | `True` (train only) | With seed_worker, shuffle is deterministic |

## Known Nondeterministic Operations

| Operation | Deterministic? | Mitigation |
|-----------|----------------|------------|
| Transformer attention (FP16) | Partially | Use FP32 or deterministic algorithms |
| Ledoit-Wolf covariance | ✅ | Uses sklearn, deterministic with fixed data |
| Mixup augmentation | With seed | Controlled by seed |
| SpecAugment (time/freq mask) | With seed | Controlled by seed |

## Environment Capture

`train.py` now writes `artifacts/experiment_provenance.json` on every run:

```json
{
  "experiment_id": "train_20260721_HHMMSS",
  "git_commit": "<commit hash>",
  "python_version": "3.11.x",
  "pytorch_version": "2.x",
  "config": { "backbone": "...", "seed": 42, ... }
}
```

## Manifest Provenance

| File | Checksum | Status |
|------|----------|--------|
| `metadata/dataset_manifest.csv` | SHA-256 in `metadata/dataset_manifest.sha256` | VERIFIED |
| Manifest validated on load | `split_utils.load_manifest_split()` | VERIFIED |

## Configuration Provenance

`config.py` is the single source for all hyperparameters. The configuration is:

- `DataConfig`: dataset path, audio parameters, augmentation
- `ModelConfig`: backbone, temporal module, embedding dimension
- `TrainingConfig`: optimizer, scheduler, batch size, seed
- `InferenceConfig`: fusion weights, risk thresholds

## Exact Reproduction Commands

```bash
# 1. Set deterministic seed (in config.py)
#    training.random_seed = 42
#    training.deterministic_cudnn = True

# 2. Verify manifest integrity
python _audit_check.py

# 3. Train
python train.py
# Output: checkpoints/best_model.pt, artifacts/experiment_provenance.json

# 4. Calibrate
python calibrate.py
# Output: checkpoints/detector_calibration.pt

# 5. Evaluate on validation (selects threshold)
python evaluate.py --split validation
# Output: checkpoints/eval_report.json, artifacts/threshold_metadata.json

# 6. Evaluate on test (uses frozen threshold)
python evaluate.py --split test
# Output: checkpoints/eval_report_test.json

# 7. Run baselines
python scripts/run_baselines.py
# Output: reports/baseline_comparison.json
```

## Expected Outputs

| Artifact | Contents |
|----------|----------|
| `checkpoints/best_model.pt` | Model state dict, optimizer, scheduler, metrics |
| `checkpoints/detector_calibration.pt` | μ, σ per signal, reference embeddings, covariance |
| `artifacts/experiment_provenance.json` | Git commit, Python/PyTorch versions, config snapshot |
| `artifacts/threshold_metadata.json` | Frozen threshold, selection method, selected_on split |
| `checkpoints/eval_report_test.json` | All metrics on test split |

## Minimum Evidence for "Reproducible"

A run is considered REPRODUCIBLE when:

1. Same seed → same manifest checksum → same splits
2. Same seed → same initialization → same loss trajectory (within FP16 tolerance)
3. Two independent runs produce metrics within bootstrap CI ranges
4. All provenance files match git commit
5. No uncommitted changes in source files

## Current Reproducibility Status

| Requirement | Status |
|-------------|--------|
| Seed control infrastructure | ✅ IMPLEMENTED |
| Deterministic cuDNN | ✅ CONFIGURED |
| Provenance tracking | ✅ IMPLEMENTED |
| Manifest checksum validation | ✅ VERIFIED |
| Two-run reproducibility test | ❌ NOT VERIFIED (needs dataset) |
| Hardware-independent reproducibility | ❌ NOT VERIFIED (needs multi-machine test) |

---

*Reproducibility infrastructure is complete. Actual reproduction requires: dataset at configured path, identical hardware (or deterministic fallback), and two independent runs.*
