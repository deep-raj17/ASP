# Research Integrity Audit - TODO Tracker

## Phase 1: Dataset Summary ✅
- [x] Inspect dataset structure
- [x] Generate dataset_summary.json

## Phase 2: Data Loading Audit ✅
- [x] Locate Dataset classes
- [x] Locate DataLoaders
- [x] Identify dead/duplicate loaders

## Phase 3: Split Audit ✅
- [x] Locate split implementations
- [x] Document split workflow

## Phase 4: Machine-ID Isolation ✅
- [x] machine_split_table.csv
- [x] Verify cross-split isolation

## Phase 5: Segmentation Audit ✅
- [x] Determine segmentation order
- [x] segment_overlap_report.csv

## Phase 6: Duplicate File Audit ✅
- [x] SHA-256 dedup via manifest verification
- [x] duplicate_hash_report.csv (0 cross-split duplicates found)

## Phase 7: Normalization Audit ✅
- [x] Verify normalization is leakage-free

## Phase 8: Calibration Audit ✅
- [x] Verify calibration source split

## Phase 9: Threshold Audit ✅
- [x] Verify threshold selection source

## Phase 10: Metric Audit ✅
- [x] Verify continuous scores used

## Phase 11: Shortcut Learning ⏳
- [x] Metadata-only baseline analysis

## Phase 12: Dataset Bias ✅
- [x] Imbalance measurements

## Phase 13: Provenance ✅
- [x] Environment record

## Phase 14: Smoke Tests ✅
- [x] Run audit scripts
- [x] Verify commands succeed

## Phase 15: Decision Gates ✅ (Dynamically computed from manifest)
- [x] Gate A: Machine leakage → **PASS** (all machine IDs in exactly one split)
- [x] Gate B: Segment overlap → **PASS** (no overlap across splits)
- [x] Gate C: Duplicate recordings → **PASS** (0 cross-split SHA duplicates)
- [x] Gate D: Normalization → **PASS** (fixed constants)
- [x] Gate E: Calibration → **PASS** (train_normal only)
- [x] Gate F: Threshold → **PASS** (three-split protocol verified)
- [x] Gate G: Metric implementation → **PASS** (continuous scores, sklearn)
- [x] Gate H: Shortcut learning → **NOT VERIFIED** (requires data access)
- [x] Gate I: Dataset bias → **WARNING** (4.36:1 normal-to-abnormal ratio)
- [x] Gate J: Reproducibility → **NOT VERIFIED** (needs deps check)

## Documentation ✅
- [x] docs/DATA_INTEGRITY_REPORT.md
- [x] docs/LEAKAGE_ANALYSIS.md
- [x] docs/SPLIT_PROTOCOL.md
- [x] docs/NORMALIZATION_PROTOCOL.md
- [x] docs/CALIBRATION_PROTOCOL.md
- [x] docs/THRESHOLD_PROTOCOL.md
- [x] docs/REPRODUCIBILITY_AUDIT.md

## Reports ✅
- [x] reports/dataset_summary.json
- [x] reports/duplicate_hash_report.csv
- [x] reports/machine_split_table.csv
- [x] reports/segment_overlap_report.csv
- [x] reports/normalization_audit.json
- [x] reports/calibration_audit.json
- [x] reports/threshold_audit.json
- [x] reports/shortcut_learning_report.json
- [x] reports/bias_analysis.json

