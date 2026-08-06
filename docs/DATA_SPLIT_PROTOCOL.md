# Data Split Protocol

## Overview

This document describes the data split protocol used in the acoustic anomaly detection project.

## Dataset Structure

```
E:\MIMII\
├── -6_dB_fan\
│   └── fan\
│       ├── id_00\
│       │   ├── normal\*.wav
│       │   └── abnormal\*.wav
│       ├── id_02\
│       ├── id_04\
│       └── id_06\
├── 0_db_fan\
├── 6_dB_fan\
├── -6_dB_pump\
├── 0_db_pump\
├── 6_dB_pump\
├── -6_dB_slider\
├── 0_db_slider\
├── 6_dB_slider\
├── -6_dB_valve\
├── 0_dB_valve\
└── 6_dB_valve\
```

## Split Assignment Method

**Algorithm:** Deterministic hash-based split

**Implementation:** `data/dataset.py` lines 136-144

```python
rel = os.path.relpath(fp, root)
key = f"{self.dcfg.split_seed}|{os.path.normpath(rel).lower()}"
h = int(hashlib.md5(key.encode()).hexdigest(), 16)
is_val = (h % 10_000) < int(self.dcfg.val_fraction * 10_000)
```

**Parameters:**
- `split_seed`: 42
- `val_fraction`: 0.15 (15%)

**Properties:**
- Deterministic: Same file always assigned to same split
- Stable across runs: Uses MD5 hash of path + seed
- File-level: Split assigned before any processing

## Split Protocol Type

**Protocol:** `machine_independent`

**Definition:** Machine IDs may appear in both train and validation splits.

**Rationale:** The current implementation does not enforce machine ID isolation. The same machine ID (e.g., id_00) can have recordings in both train and validation splits.

**Machine ID Overlap:**
- id_00: appears in both train and val
- id_02: appears in both train and val
- id_04: appears in both train and val
- id_06: appears in both train and val

**Implementation:** machine IDs are assigned to disjoint train/validation/test buckets so that the experiment evaluates unseen machine IDs rather than repeated ones.

## Split Statistics

### Overall Distribution
- **Total Files:** 53,046
- **Train:** 45,222 (85.2%)
- **Validation:** 7,824 (14.8%)
- **Test:** 0 (0%)

### Label Distribution
- **Normal:** 43,146 (81.3%)
- **Abnormal:** 9,900 (18.7%)

### Per-Split Label Distribution

**Train:**
- Normal: 36,827
- Abnormal: 8,395

**Validation:**
- Normal: 6,319
- Abnormal: 1,505

### Per-Machine-Type Distribution

| Machine Type | Train | Validation | Total |
|-------------|-------|------------|-------|
| fan | 13,350 | 2,289 | 15,639 |
| pump | 10,726 | 1,889 | 12,615 |
| slider | 10,472 | 1,810 | 12,282 |
| valve | 10,674 | 1,836 | 12,510 |

### Per-Noise-Condition Distribution

| Noise Condition | Train | Validation | Total |
|----------------|-------|------------|-------|
| -6_dB | 15,313 | 2,706 | 18,019 |
| 0_dB | 14,537 | 2,471 | 17,008 |
| 6_dB | 15,372 | 2,647 | 18,019 |

## Critical Issue: No Test Set

**Status:** The project currently has NO separate test set.

**Impact:**
- All reported metrics are validation metrics
- No untouched test set for final evaluation
- Cannot claim generalization to unseen data
- Threshold selection and evaluation both use validation set

**Recommendation:** Create a proper train/validation/test split with:
- Train: 70% for model training
- Validation: 15% for hyperparameter tuning and threshold selection
- Test: 15% for final evaluation only

## Data Leakage Audit Results

**Audit Status:** PASSED

**Checks Performed:**
- ✓ No duplicate checksums across splits
- ✓ No unknown split values
- ✓ No unknown machine IDs
- ✓ No unknown labels
- ✓ Machine ID overlap is expected for machine_dependent protocol

**Audit Report:** `reports/data_leakage_audit.json`

## Manifest

**File:** `metadata/dataset_manifest.csv`

**Columns:**
- file_id
- relative_path
- absolute_path
- noise_condition
- machine_type
- machine_id
- label
- split
- source_recording
- segment_start
- segment_end
- duration_seconds
- sample_rate
- num_frames
- num_channels
- file_size_bytes
- sha256

**Checksum:** `metadata/dataset_manifest.sha256`

## Reproducibility

To reproduce the exact same split:

1. Use `split_seed = 42`
2. Use `val_fraction = 0.15`
3. Use the MD5 hash-based assignment algorithm
4. Apply to the same dataset root

The split is deterministic and will be identical across machines and runs.
