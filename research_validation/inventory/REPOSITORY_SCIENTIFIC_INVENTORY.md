# Repository Scientific Inventory

**Inventory date:** 2026-07-24  
**Branch:** `blackboxai/research-integrity-audit`  
**Observed commit:** `3b78096e6ffcfb7f6ebff5fd6705f6b75124c2c7`  
**Working tree:** DIRTY; existing user changes preserved

## Located components

| Area | Evidence | Status |
|---|---|---|
| Data and splits | `metadata/dataset_manifest.csv`, `data/dataset.py`, `utils/split_utils.py` | PRESENT; PMPS provenance blocker remains |
| Training | `train.py`, `training/`, `artifacts/EXP-CHAAD-001/` | PRESENT; underfitting diagnosis documented |
| Evaluation | `evaluate.py`, `scripts/audit_evaluation_pipeline.py` | PRESENT; corrected validation export verified |
| Leakage audits | `scripts/audit_data_leakage.py`, `reports/data_leakage_audit.json` | PRESENT |
| Baseline infrastructure | `scripts/run_baselines.py`, `artifacts/baselines/` | PRESENT; publication certification not complete |
| Statistical infrastructure | `scripts/statistical_validation.py` | PRESENT; complete evidence not available |
| ROS governance | `ros/`, `projects/chaad/`, `publication/*_GATE*` | PRESENT; downstream gates blocked |
| Tests | `tests/`, `tests/ros/` | PRESENT |

## Scope decision

Only repository discovery was executed. No retraining, test-set model
selection, raw-data mutation, manuscript drafting, or publication packaging
was performed.
