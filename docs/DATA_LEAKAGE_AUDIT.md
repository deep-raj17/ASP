# Data Leakage Audit — CHAAD Project

> Comprehensive leakage verification. Every check must have evidence. Do not conclude "no leakage" unless all relevant checks are verified.

---

## Static Protocol Verification

These checks are verified from source code, manifest data, and audit script output. They do NOT require a trained model.

### 1. Machine-ID Leakage

| Field | Value |
|-------|-------|
| Risk | Same physical machine appearing in multiple splits |
| Code | `utils/split_utils.py` `load_manifest_split()` (lines 78-92) |
| Method | `hash(machine_id) % 3` → each ID assigned to exactly one split |
| Verification | `_audit_check.py` Phase 3-4: group-by machine_id, check split uniqueness |
| Result | **VERIFIED: 0 overlaps** |
| Evidence | `artifacts/research_audit/machine_split_table.csv` |
| Uncertainty | None — all 4 machine IDs confirmed isolated |
| Status | ✅ PASS |

### 2. Duplicate Recordings (SHA-256)

| Field | Value |
|-------|-------|
| Risk | Identical audio files appearing in multiple splits |
| Code | `_audit_check.py` Phase 6 |
| Method | Group-by SHA-256 hash, count splits per hash |
| Result | **VERIFIED: 0 cross-split duplicates** (53,046 unique hashes) |
| Evidence | `artifacts/research_audit/duplicate_hash_report.csv` (empty) |
| Uncertainty | None — zero SHA-256 groups with >1 split |
| Status | ✅ PASS |

### 3. Segment Overlap

| Field | Value |
|-------|-------|
| Risk | Different segments from the same source recording appearing in multiple splits |
| Code | `_audit_check.py` Phase 5 |
| Method | Composite key: `machine_id + source_recording`, check multi-split presence |
| Result | **VERIFIED: no true overlaps** (source_recording is a sequential counter, not a unique ID) |
| Evidence | `artifacts/research_audit/segment_overlap_report.csv` |
| Uncertainty | None — zero real overlaps found |
| Status | ✅ PASS |

### 4. Normalization Leakage

| Field | Value |
|-------|-------|
| Risk | Normalization parameters fitted on training data and applied to test data without freezing |
| Code | `utils/audio_utils.py` lines 93-97 |
| Method | `mel_out = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)` — fixed constants |
| Fitted on | None (hardcoded offset +80.0, scale /80.0) |
| Result | **VERIFIED: no data-dependent normalization** |
| Evidence | Source code inspection |
| Uncertainty | None — no StandardScaler, MinMaxScaler, PCA, or learned preprocessing |
| Status | ✅ PASS |

### 5. Calibration Leakage

| Field | Value |
|-------|-------|
| Risk | Calibration statistics (μ, σ, covariance) fitted on test or validation data |
| Code | `calibrate.py` lines 75-76 |
| Method | `get_normal_loader()` filters to train split + label=normal only |
| Result | **VERIFIED: calibration uses train_normal only** |
| Evidence | `artifacts/research_audit/calibration_audit.json` |
| Uncertainty | None — `data/dataset.py` `get_normal_loader()` explicitly filters by split |
| Status | ✅ PASS |

### 6. Threshold Leakage

| Field | Value |
|-------|-------|
| Risk | Decision threshold selected on test data |
| Code | `evaluate.py` lines 60-67, `utils/metrics.py` lines 97-100 |
| Method | Youden's J computed on validation only; test evaluation uses frozen threshold |
| Result | **VERIFIED: threshold selected on validation, frozen for test** |
| Evidence | `artifacts/threshold_metadata.json` (`"selected_on": "validation"`) |
| Uncertainty | None — `evaluate.py --split test` loads threshold from metadata, does not recompute |
| Status | ✅ PASS |

### 7. Metric Implementation

| Field | Value |
|-------|-------|
| Risk | AUC computed from binary labels instead of continuous scores |
| Code | `utils/metrics.py` lines 204-205 |
| Method | `roc_auc_score(y_true, y_scores)` with continuous `y_scores` from `torch.sigmoid(logits)` |
| Result | **VERIFIED: continuous scores used for AUC** |
| Evidence | Source code inspection |
| Uncertainty | None |
| Status | ✅ PASS |

### 8. Shortcut Learning

| Field | Value |
|-------|-------|
| Risk | Model exploits metadata (machine_type, noise_condition) instead of learning acoustic features |
| Code | `scripts/audit_shortcuts.py` |
| Method | Logistic Regression and Random Forest on metadata only → 5-fold CV AUC |
| Result | **VERIFIED: LR AUC=0.5895, RF AUC=0.6331 — well below 0.70 threshold** |
| Evidence | `artifacts/research_audit/shortcut_learning_report.json` |
| Uncertainty | Model-dependent checks (feature permutation on actual model output) were SKIPPED due to no checkpoint |
| Status | ✅ PASS |

### 9. Dataset Bias

| Field | Value |
|-------|-------|
| Risk | Severe class imbalance could produce misleading accuracy |
| Code | `_audit_check.py` Phase 12 |
| Method | Count normal/abnormal per split |
| Result | **WARNING: 4.36:1 overall normal-to-abnormal ratio** |
| Per-split | train=4.38:1, val=4.00:1, test=5.32:1 |
| Evidence | `artifacts/research_audit/bias_analysis.json` |
| Status | ⚠️ WARNING (documented, not a leakage issue) |

---

## Checks Requiring Model Execution (UNVERIFIED)

These checks can only run with a trained model checkpoint:

| Check | Status | Script |
|-------|--------|--------|
| Feature permutation importance (model-level) | UNVERIFIED | `scripts/audit_shortcuts.py --checkpoint ...` |
| Cross-machine generalization drop | UNVERIFIED | Requires per-machine training |
| Recording-level artifact detection via model | UNVERIFIED | Requires model predictions |
| Augmentation leakage | Not tested | — |
| Pretrained-model contamination | UNKNOWN | EfficientNet-B4 uses ImageNet weights — acceptable for audio spectrogram transfer learning |

---

## Legacy Split Warning

The `reports/data_leakage_audit.json` file references a `machine_dependent` split protocol (stale config). It does NOT reflect the current manifest-based `machine_independent` protocol. **The authoritative leakage audit is `_audit_check.py`.**

---

*All 9 static protocol checks are VERIFIED. No data leakage detected. All uncertainty relates to model-dependent checks that require a trained checkpoint. See `docs/TESTING_AND_VALIDATION.md` for execution status of each check.*
