# Dataset and Splits — CHAAD Project

> Last updated: 2026-07-21

## Dataset Identity

| Field | Value |
|-------|-------|
| Dataset name | MIMII (Malfunctioning Industrial Machine Investigation and Inspection) |
| Source | MIMII research dataset |
| License | UNKNOWN (requires confirmation) |
| Version | UNKNOWN |
| Local path (configured) | `E:\MIMII` |
| Total WAV files | 53,046 (from manifest) |

## Dataset Composition

| Dimension | Values | Count |
|-----------|--------|-------|
| Machine types | fan, pump, slider, valve | 4 |
| Machine IDs | id_00, id_02, id_04, id_06 | 4 |
| Noise conditions | -6_dB, 0_dB, 6_dB | 3 |
| Labels | normal, abnormal | 2 |
| Total files | 53,046 | — |

### Per-Machine Breakdown (from manifest)

| Machine ID | Split | Total | Normal | Abnormal |
|-----------|-------|-------|--------|----------|
| id_04 | train | 12,045 | 9,807 | 2,238 |
| id_00 | val | 14,292 | 11,217 | 3,075 |
| id_02 | val | 13,962 | 11,391 | 2,571 |
| id_06 | test | 12,747 | 10,731 | 2,016 |

### Per-Machine-Type Breakdown

| Type | Total | Abnormal | Ratio |
|------|-------|----------|-------|
| fan | 15,639 | 4,425 | 28.3% |
| pump | 12,615 | 1,368 | 10.8% |
| slider | 12,282 | 2,670 | 21.7% |
| valve | 12,510 | 1,437 | 11.5% |

### Class Imbalance

| Split | Normal | Abnormal | Ratio |
|-------|--------|----------|-------|
| train | 9,807 | 2,238 | 4.38:1 |
| val | 22,608 | 5,646 | 4.00:1 |
| test | 10,731 | 2,016 | 5.32:1 |
| **Overall** | **43,146** | **9,900** | **4.36:1** |

## Expected Directory Layout

```
E:\MIMII\
  ├── -6_dB_fan\
  │   └── fan\
  │       ├── id_00\normal\*.wav
  │       ├── id_00\abnormal\*.wav
  │       ├── id_02\...
  │       ├── id_04\...
  │       └── id_06\...
  ├── -6_dB_pump\...
  ├── -6_dB_slider\...
  ├── -6_dB_valve\...
  ├── 0_dB_fan\...
  ├── 0_dB_pump\...
  ├── 0_dB_slider\...
  ├── 0_dB_valve\...
  ├── 6_dB_fan\...
  ├── 6_dB_pump\...
  ├── 6_dB_slider\...
  └── 6_dB_valve\...
```

## Authoritative Manifest

| File | Purpose | Status |
|------|---------|--------|
| `metadata/dataset_manifest.csv` | Single source of truth for splits | VERIFIED |
| `metadata/dataset_manifest.sha256` | Integrity checksum | VERIFIED |

Manifest columns (from `utils/split_utils.py` `REQUIRED_COLUMNS`):
`file_id`, `relative_path`, `absolute_path`, `noise_condition`, `machine_type`, `machine_id`, `label`, `split`, `source_recording`, `segment_start`, `segment_end`, `duration_seconds`, `sample_rate`, `num_frames`, `num_channels`, `file_size_bytes`, `sha256`

## Split Protocol

| Protocol | Machine-Independent |
|----------|---------------------|
| Method | `hash(machine_id) % 3` assignment |
| Split seed | N/A (deterministic from machine_id only) |
| Cross-validation | None (single fixed split) |
| Group | Machine ID (each ID in exactly one split) |

### Split Assignment

```
id_04 → train  (bucket 0)
id_00 → val    (bucket 1)
id_02 → val    (bucket 1)
id_06 → test   (bucket 2)
```

## Data Integrity Verifications

### Machine-ID Isolation

| Check | Method | Result |
|-------|--------|--------|
| Cross-split machine ID overlap | Group-by analysis | VERIFIED: 0 overlaps |
| Each ID in one split | manifest `split` column | VERIFIED |

### Duplicate Detection

| Check | Method | Result |
|-------|--------|--------|
| SHA-256 duplicates | Group-by hash, check cross-split | VERIFIED: 0 cross-split duplicates |
| Segment overlap | Composite key (machine_id + source_recording) | VERIFIED: no true overlaps |

### Manifest Integrity

| Check | Method | Result |
|-------|--------|--------|
| Manifest checksum | SHA-256 of CSV file | VERIFIED |
| Required columns present | `split_utils.py` validation | VERIFIED |
| No unknown split values | Enum validation | VERIFIED |

## Calibration Subset

| Subset | Source | Samples |
|--------|--------|---------|
| train_normal | train split, label=normal only | 9,807 (from manifest) |
| Reference pool | train_normal | 37,685 (from legacy calibration report — UNVERIFIED discrepancy) |

**Note**: The legacy calibration report references 37,685 normal samples. The current manifest shows 9,807 normal training samples. This discrepancy suggests the legacy report used a different split protocol. **The correct reference pool size under the current split is 9,807**.

## Threshold Selection Subset

| Split | Used for | Samples |
|-------|----------|---------|
| val | Threshold selection (Youden's J) | 28,254 |

## Untouched Test Rule

The **test split (id_06, 12,747 samples)** must remain untouched during:
- Model training
- Hyperparameter tuning
- Calibration
- Threshold selection
- Any ablation or model selection decision

Test metrics are computed exactly once with a frozen validation threshold.

## Known Dataset-Access Uncertainty

- **UNKNOWN**: Whether `E:\MIMII` is currently accessible on this machine
- **UNKNOWN**: Whether the actual WAV files correspond to the manifest entries
- **UNKNOWN**: Dataset version and exact source URL

---

*Data sourced from `metadata/dataset_manifest.csv` (verified), `_audit_check.py` output (verified), and code inspection (verified). Calibration discrepancy noted.*
