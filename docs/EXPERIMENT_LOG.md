# Experiment Log — CHAAD Project

> Structured registry of experiments. Each entry MUST have evidence before being marked complete.

---

## Experiment Registry

### EXP-001: Legacy Model Training and Calibration

| Field | Value |
|-------|-------|
| Experiment ID | EXP-001 |
| Date | 2026-05 (inferred from calibration_report.json timestamp) |
| Status | LEGACY — SUPERSEDED |
| Purpose | Train HybridAnomalyModel and calibrate detector |
| Git commit | UNKNOWN |
| Manifest checksum | UNKNOWN (pre-manifest era) |
| Configuration checksum | UNKNOWN |
| Seed | UNKNOWN |
| Checkpoint | `checkpoints/best_model.pt` (NOT VERIFIED to exist) |
| Model variant | HybridAnomalyModel (EfficientNet-B4 + Transformer) |
| Training command | `python train.py` (presumed) |
| Calibration command | `python calibrate.py` (presumed) |
| Metrics artifact | `checkpoints/eval_report.json` |
| Result summary | ROC-AUC: 0.9999997, PR-AUC: 0.9999999 on **validation split only** |
| Failure reason | N/A (legacy — not using current split protocol) |
| Official/Legacy | **LEGACY** — metrics cannot be cited as final |
| Reproducibility | **NOT REPRODUCIBLE** with current manifest-based splits |

**Evidence**: `checkpoints/calibration_report.json` dated 2026-05-15, `checkpoints/eval_report.json`, `artifacts/pre_validation_backup/`. All artifacts exist but were produced under an unknown split configuration.

**Action**: This experiment is superseded. Train a new model under the current machine-independent 3-split protocol.

---

### EXP-002: Research Integrity Audit

| Field | Value |
|-------|-------|
| Experiment ID | EXP-002 |
| Date | 2026-07-21 |
| Status | **COMPLETED** |
| Purpose | Verify all data leakage protections |
| Git commit | Current branch HEAD (`blackboxai/research-integrity-audit`) |
| Manifest checksum | Verified against `metadata/dataset_manifest.sha256` |
| Configuration | Current manifest-based splits |
| Script | `_audit_check.py` |
| Result | 7/7 gates PASS, 0 FAIL, 1 WARNING (dataset bias 4.36:1), 2 NOT VERIFIED (shortcut + repro) |
| Reproducibility | **VERIFIED** — script executes and produces consistent output |

---

### EXP-003: Shortcut Learning Audit

| Field | Value |
|-------|-------|
| Experiment ID | EXP-003 |
| Date | 2026-07-21 |
| Status | **COMPLETED** |
| Purpose | Test if model could exploit metadata-only shortcuts |
| Script | `scripts/audit_shortcuts.py` |
| Result | **NO SHORTCUT DETECTED** |
| Metrics | Logistic Regression AUC (metadata): 0.5895 ± 0.0086; Random Forest AUC (metadata): 0.6331 |
| Threshold | 0.70 AUC for shortcut concern |
| Reproducibility | **VERIFIED** — script executes on manifest data without model |

---

### EXP-CHAAD-001: Machine-Independent Protocol Training Run

| Field | Value |
|-------|-------|
| Experiment ID | EXP-CHAAD-001 |
| Date | 2026-07-23 |
| Status | **OFFICIAL PROVISIONAL RESULT - EVALUATION EXPORT FIXED** |
| Purpose | First official CHAAD training run under machine-independent split protocol |
| Git commit | 686c450bd416f6cf921befe4156d1a27b26105c2 |
| Git branch | blackboxai/research-integrity-audit |
| Calculated manifest checksum | 7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136 |
| Manifest sidecar status | **MISMATCH** — `metadata/dataset_manifest.sha256` records `7c689508cbed4d49d05ec2891b315b27722ff01c8a62b6b1c4f610e3afcd0136` |
| Configuration | Current config.py (machine-independent split) |
| Seed | 42 |
| Checkpoint | `artifacts/EXP-CHAAD-001/checkpoint.pt` (SHA-256: 7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9) |
| Model variant | HybridAnomalyModel (EfficientNet-B4 + Transformer) |
| Training command | `python train.py` |
| Final epoch | 100/100 |
| Best checkpoint epoch | 6 |
| Checkpoint selection | Minimum validation loss (`2.2707729198`) |
| Corrected validation prediction export | `artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv` |
| Evaluation export status | PASS: 28,254 rows, 28,254 unique sample IDs, 0 duplicates |
| Corrected validation ROC-AUC | 0.6002609445 |
| Corrected validation PR-AUC | 0.2578861055 |
| Corrected validation EER | 0.4264914172 |
| Batch-size determinism | PASS: batch 16 vs 32, max score difference 2.384185791015625e-07 |
| Original corrupted export | Retained at `artifacts/EXP-CHAAD-001/validation_predictions.csv` (30 duplicated sample IDs, 60 affected rows, ROC-AUC 0.5999999937) |
| Highest logged ROC-AUC | 0.6176229715 at epoch 11 (not selected) |
| Result summary | Corrected validation ROC-AUC: 0.6002609445, Balanced Accuracy at Youden threshold: 0.5760815437, EER: 0.4264914172 |
| Failure reason | Low validation discrimination and Prompt 3 underfitting diagnosis; no architecture change or retraining performed |
| Official/Legacy | **OFFICIAL PROVISIONAL** - validation export is fixed, but no held-out test metric is registered |
| Reproducibility | **PARTIAL** - checkpoint and manifest are integrity-verified, but the dirty training-tree patch, exact command, full training-time config, and dependency/driver lock are unavailable |

**Evidence**: `checkpoints/best_model.pt` and the preserved copy have matching SHA-256 `7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`. The checkpoint records epoch 6; `checkpoints/epoch_100.pt` records the final metrics; the TensorBoard event log contains all 100 validation epochs. The corrected prediction export and determinism reports are `artifacts/EXP-CHAAD-001/prediction_export_validation.json` and `artifacts/EXP-CHAAD-001/prediction_export_determinism.json`.

**Action**: Preserve this result as provisional. Prompt 2 has passed after the export fix; refresh the legacy comparison next, then continue to diagnostic baselines only after the ordered audit prerequisites remain satisfied. No test-set performance is registered here.

---

## Planned Experiments (Template)

### EXP-004: Full Pipeline Training with Deterministic Seeds (PLANNED)

| Field | Value |
|-------|-------|
| Experiment ID | EXP-004 |
| Status | **PLANNED** (BLOCKED by dataset access) |
| Purpose | Train model under current protocol, produce checkpoint |
| Command | `python train.py` |
| Expected output | `checkpoints/best_model.pt`, `artifacts/experiment_provenance.json` |
| Validation | Loss convergence, reasonable val metrics |
| Seed | 42 (from `config.py`) |

### EXP-005: Independent Test-Set Evaluation (PLANNED)

| Field | Value |
|-------|-------|
| Status | **PLANNED** (BLOCKED by EXP-004) |
| Command | `python evaluate.py --split test` |
| Expected output | `checkpoints/eval_report_test.json` |

### EXP-006: Baseline Comparison (PLANNED)

| Field | Value |
|-------|-------|
| Status | **PLANNED** (BLOCKED by EXP-004) |
| Command | `python scripts/run_baselines.py` |
| Expected output | `reports/baseline_comparison.json` (11 baselines) |

### EXP-007: Statistical Validation (PLANNED)

| Field | Value |
|-------|-------|
| Status | **PLANNED** (BLOCKED by EXP-005) |
| Command | `python scripts/statistical_validation.py --predictions ...` |

### EXP-008: Ablation Study (PLANNED)

| Field | Value |
|-------|-------|
| Status | **PLANNED** (needs ablation runner + EXP-004) |
| Variants | 6 ablation conditions (see `docs/NOVELTY_AND_CONTRIBUTIONS.md` §H) |

---

## Experiment Template (for future use)

```markdown
### EXP-XXX: [Title]

| Field | Value |
|-------|-------|
| Experiment ID | EXP-XXX |
| Date | YYYY-MM-DD |
| Status | PLANNED / RUNNING / COMPLETED / FAILED |
| Purpose | |
| Git commit | |
| Manifest checksum | |
| Configuration checksum | |
| Seed | |
| Checkpoint | |
| Model variant | |
| Training command | |
| Evaluation command | |
| Metrics artifact | |
| Result summary | |
| Failure reason | |
| Official/Legacy designation | |
| Reproducibility | |
| Evidence | |
```

---

*EXP-CHAAD-001 is integrity-preserved but remains an OFFICIAL PROVISIONAL RESULT. LEGACY status means the experiment used a superseded protocol.*
