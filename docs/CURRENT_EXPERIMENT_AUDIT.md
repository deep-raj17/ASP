# Current Experiment Audit

**Generated:** 2026-07-20T15:21:00+05:30  
**Git Commit:** 686c450bd416f6cf921befe4156d1a27b26105c2  
**Branch:** research-validation

## Executive Summary

The project reports unusually high performance metrics:
- ROC-AUC: 99.99997%
- PR-AUC: 99.99986%
- Accuracy: 99.96%
- Precision: 99.80%
- Recall: 100%
- F1: 99.90%

This audit documents the experimental protocol to determine scientific validity.

---

## 1. Data Split Protocol

**File:** `data/dataset.py` (lines 70-161)

**Method:** Deterministic hash-based split
- Uses MD5 hash of `split_seed + relative_path` (line 137-138)
- Validation fraction: 15% (config.py line 26)
- Split seed: 42 (config.py line 29)
- Split is applied at **file level**, not segment level

**Code:**
```python
key = f"{self.dcfg.split_seed}|{os.path.normpath(rel).lower()}"
h = int(hashlib.md5(key.encode()).hexdigest(), 16)
is_val = (h % 10_000) < int(self.dcfg.val_fraction * 10_000)
```

**Critical Finding:** No explicit test split exists. The project has only train/val splits.

**DataLoader Factory:** `data/dataset.py` (lines 217-245)
- Returns `train_loader` and `val_loader` only
- No `test_loader` defined

**Evaluation Target:** `evaluate.py` (line 46)
- Evaluates on `split="val"` 
- No separate test set evaluation

---

## 2. Segmentation Protocol

**File:** `utils/audio_utils.py` (lines 117-124)

**Method:** Fixed-duration padding/trimming
- Target duration: 10 seconds (config.py line 33)
- Sample rate: 16,000 Hz (config.py line 32)
- Target length: 160,000 samples

**Code:**
```python
def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    length = waveform.shape[-1]
    if length > target_length:
        return waveform[..., :target_length]
    elif length < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - length))
    return waveform
```

**Spectrogram Parameters:** `config.py` (lines 36-41)
- n_fft: 2048
- hop_length: 512
- n_mels: 128
- fmin: 20.0 Hz
- fmax: 8000.0 Hz

**Critical Finding:** No sliding window segmentation. Each file is treated as a single sample after padding/trimming.

**Order of Operations:**
1. Files discovered via glob (dataset.py line 78)
2. Split assigned via hash (dataset.py line 139)
3. Files loaded and padded/trimmed on-the-fly (dataset.py line 212)

**Assessment:** Safe order - split assignment happens before any segmentation.

---

## 3. Normalization Protocol

**File:** `utils/audio_utils.py` (lines 93-97)

**Method:** Fixed-scale normalization to [0,1]

**Code:**
```python
if self.cfg.normalize_mel:
    mel_out = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
```

**Type:** Per-file normalization with fixed constants
- Uses fixed offset (+80.0) and scale (/80.0)
- Not fitted on any data
- Constants are hardcoded, not learned

**Critical Finding:** This is safe - no global statistics are fitted on training data.

**No Global Transformations Found:**
- No StandardScaler
- No MinMaxScaler with fit_transform
- No PCA
- No imputation
- No learned preprocessing

---

## 4. Calibration Protocol

**File:** `calibrate.py` (lines 40-98)

**Method:** Fit reference distribution on normal training samples

**Code:**
```python
normal_loader = get_normal_loader(cfg)  # Returns only normal train samples
detector.fit_reference_distribution(normal_loader)
```

**Normal Loader:** `data/dataset.py` (lines 250-264)
- Loads full train dataset
- Filters to keep only `label == 0` (normal) samples
- Uses train split only

**Calibration Statistics Computed:** `inference/detector.py` (lines 84-152)
- Reconstruction error: μ, σ
- Embedding cosine distance: μ, σ
- Mahalanobis distance: μ, σ
- Contrastive distance: μ, σ
- Reference embedding mean
- Reference covariance matrix (Ledoit-Wolf)
- Reference embedding pool (for k-NN)

**Calibration Report:** `checkpoints/calibration_report.json`
- Generated: 2026-05-15T10:46:04
- Reference pool size: 37,685 samples
- All statistics fitted on train_normal only

**Critical Finding:** Calibration uses train_normal only - SAFE.

---

## 5. Threshold Selection Protocol

**File:** `utils/metrics.py` (lines 97-100)

**Method:** Youden's J statistic on ROC curve

**Code:**
```python
fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_scores)
j_scores = tpr_arr - fpr_arr
best_thresh = float(thresholds[np.argmax(j_scores)])
```

**Application:** `evaluate.py` (lines 45-82)
- Computes metrics on validation set
- Threshold selected on validation scores and validation labels
- Current threshold: 0.313720703125 (from eval_report.json)

**Critical Finding:** Threshold is selected on validation set, which is correct for validation evaluation. However, there is NO separate test set to apply this frozen threshold to.

**No Threshold Storage:** No explicit `threshold_metadata.json` exists. Threshold is embedded in eval_report.json but not frozen separately.

---

## 6. Metric Calculation Protocol

**File:** `utils/metrics.py` (lines 78-128)

**ROC-AUC Calculation:** Line 118
```python
roc_auc=float(roc_auc_score(y_true, y_scores))
```

**PR-AUC Calculation:** Line 119
```python
pr_auc=float(average_precision_score(y_true, y_scores))
```

**Input:** Continuous anomaly scores (y_scores), not binary labels

**Critical Finding:** CORRECT - uses continuous scores for AUC calculation.

**Current Metrics:** `checkpoints/eval_report.json`
- ROC-AUC: 0.9999996920031703 (99.99997%)
- PR-AUC: 0.9999986781462323 (99.99986%)
- Accuracy: 0.9996239187664535 (99.96%)
- Precision: 0.9980106100795756 (99.80%)
- Recall: 1.0 (100%)
- F1: 0.9990043146365748 (99.90%)
- Confusion Matrix: [[6469, 3], [0, 1505]]
  - 6469 true negatives
  - 3 false positives
  - 0 false negatives
  - 1505 true positives

**Evaluation Set:** Validation set (7,974 samples total)

---

## 7. Task Type Classification

**Type:** Supervised Binary Classification

**Evidence:**
- Uses both normal (label=0) and abnormal (label=1) samples during training (dataset.py lines 90-96)
- Loss function includes BCE with positive class weight (config.py line 97)
- Mixup augmentation uses labels (dataset.py line 179)
- Validation metrics computed using both classes

**Not:** 
- Unsupervised anomaly detection (uses labels)
- Semi-supervised anomaly detection (uses abnormal samples in training)

**Implication:** Comparisons with published unsupervised methods would be invalid without clear disclosure of supervised setting.

---

## 8. Random Seed Usage

**Found:** `config.py` line 29
- `split_seed: int = 42` - Used for deterministic train/val split

**Not Found:**
- No `random.seed()` calls
- No `numpy.random.seed()` calls  
- No `torch.manual_seed()` calls in main training scripts

**Critical Finding:** Random seed is only used for data splitting, not for model initialization or training randomness. This means results may not be fully reproducible across runs.

---

## 9. Machine ID Isolation

**Current Protocol:** Not explicitly enforced

**Evidence:**
- Split is based on file path hash, not machine ID
- Same machine ID (e.g., id_00) can appear in both train and val splits
- No GroupShuffleSplit or GroupKFold used

**Potential Issue:** If the same machine ID appears in both train and val, this could represent data leakage if recordings from the same physical machine are split across splits.

**Needs Verification:** Audit required to check if machine IDs are disjoint across splits.

---

## 10. Critical Issues Summary

### High-Priority Issues:

1. **No Separate Test Set**
   - Evaluation is on validation set only
   - No untouched test set for final evaluation
   - Threshold selection and evaluation both use validation data
   - This is a development/validation setup, not a final test setup

2. **Machine ID Isolation Unknown**
   - Need to verify if machine IDs are disjoint across train/val
   - If same machine ID appears in both splits, this is data leakage

3. **No Multi-Seed Evaluation**
   - Only split_seed=42 used
   - No verification of stability across random seeds
   - Model training randomness not controlled

### Medium-Priority Issues:

4. **Threshold Not Frozen Explicitly**
   - Threshold embedded in eval_report.json
   - No separate threshold_metadata.json
   - No guard to prevent test data usage in threshold selection

5. **No Confidence Intervals**
   - No bootstrap confidence intervals reported
   - No uncertainty quantification

6. **No Subgroup Analysis**
   - No per-machine, per-condition, per-ID results
   - Cannot verify if high performance is uniform across subgroups

### Low-Priority Issues:

7. **Task Type Misclassification Risk**
   - Supervised method but may be compared with unsupervised baselines
   - Need clear disclosure in any publication

---

## 11. Files Referenced

- `data/dataset.py` - Data loading and split logic
- `training/trainer.py` - Training loop
- `calibrate.py` - Calibration script
- `evaluate.py` - Evaluation script
- `config.py` - Configuration
- `utils/audio_utils.py` - Audio preprocessing
- `inference/detector.py` - Anomaly detector
- `utils/metrics.py` - Metric computation
- `checkpoints/eval_report.json` - Current evaluation results
- `checkpoints/calibration_report.json` - Calibration statistics

---

## 12. Next Steps

Proceed to Phase 2: Data-Leakage Audit to verify:
1. Machine ID isolation across splits
2. No duplicate files across splits
3. No temporal overlap across splits
4. Create dataset manifest
5. Verify split protocol matches declared intent
