# Testing and Validation — CHAAD Project

> Last updated: 2026-07-21

## Testing Strategy

The project uses a multi-layer validation approach:

1. **Data integrity tests** — manifest checksums, split verification, SHA-256 dedup
2. **Audit scripts** — systematic protocol verification
3. **Unit tests** — code-level validation
4. **Model evaluation** — metric computation on held-out data
5. **Publication audit** — gate-based go/no-go assessment

## Available Test Suites

### Data Integrity Tests

| Test | File | Command | Status |
|------|------|---------|--------|
| Unit test suite | `tests/test_data_integrity.py` | `python -m pytest tests/` | UNVERIFIED (not executed) |
| Dataset verification | `verify_dataset.py` | `python verify_dataset.py` | IMPLEMENTED |

### Audit Scripts

| Audit | Script | Result | Date |
|-------|--------|--------|------|
| Research integrity audit | `_audit_check.py` | 7/7 gates PASS | 2026-07-21 |
| Shortcut learning audit | `scripts/audit_shortcuts.py` | PASS (metadata AUC=0.59) | 2026-07-21 |
| Publication go/no-go | `scripts/run_publication_audit.py` | 57.1% (CONDITIONAL) | 2026-07-21 |
| Data leakage audit | `scripts/audit_data_leakage.py` | IMPLEMENTED (requires dataset) | UNVERIFIED |
| Metric recomputation | `scripts/recompute_metrics.py` | IMPLEMENTED (requires checkpoint) | UNVERIFIED |
| Inference verification | `scripts/verify_inference.py` | IMPLEMENTED | UNVERIFIED |

### Model Evaluation

| Evaluation | Command | Expected Output | Status |
|------------|---------|-----------------|--------|
| Validation evaluation | `python evaluate.py --split val` | `checkpoints/eval_report.json` | UNVERIFIED (requires checkpoint) |
| Test evaluation | `python evaluate.py --split test` | `checkpoints/eval_report_test.json` | UNVERIFIED (requires checkpoint) |
| Baseline comparisons | `python scripts/run_baselines.py` | `reports/baseline_comparison.json` | UNVERIFIED (requires checkpoint) |
| Statistical validation | `python scripts/statistical_validation.py --predictions ...` | `reports/statistical_validation_report.json` | UNVERIFIED (requires predictions) |
| Subgroup analysis | `python scripts/evaluate_subgroups.py` | `reports/per_machine_results.csv` | VERIFIED (distribution only, no model needed) |

## Commands With Verified Results

```bash
# Research integrity audit — EXECUTED SUCCESSFULLY
python _audit_check.py
# Output: 7/7 gates PASS, 0 FAIL, 1 WARNING, 2 NOT VERIFIED (shortcut + repro)

# Shortcut learning audit — EXECUTED SUCCESSFULLY
python scripts/audit_shortcuts.py
# Output: NO SHORTCUT DETECTED, metadata AUC=0.5895 (LR), 0.6331 (RF)

# Publication go/no-go audit — EXECUTED SUCCESSFULLY
python scripts/run_publication_audit.py --verbose
# Output: CONDITIONAL, 57.1%, 5/10 PASS, 0 critical failures
```

## Tests Requiring a Model Checkpoint

All of the following require `checkpoints/best_model.pt`:

- `python evaluate.py --split test`
- `python evaluate.py --split val`
- `python scripts/run_baselines.py`
- `python scripts/statistical_validation.py`
- `python scripts/recompute_metrics.py`
- `python scripts/verify_inference.py`

## Test Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No unit test for `HybridAnomalyModel` forward pass | Cannot verify model shapes or output structure without training | Medium |
| No integration test for train→calibrate→evaluate pipeline | Cannot verify end-to-end flow without dataset | High |
| No multi-seed reproducibility test | Cannot claim deterministic results without verification | Medium |
| No regression test for metric computation | Changes to `utils/metrics.py` could silently break | Low |
| No test for edge deployment code | Edge pipeline is untested | Low |

## Definition of Done for Testing

A feature is considered TESTED when:

1. The relevant audit script executes without errors
2. Expected output artifacts are generated
3. Output values are within reasonable ranges (not NaN, not 1.0 exactly for cross-validation metrics)
4. Results are documented in this file with execution date and output summary
5. Any warnings are explained

## Failure Interpretation

| Audit failure | Meaning | Action |
|---------------|---------|--------|
| `_audit_check.py` Gate A FAIL | Machine-ID leakage detected | Regenerate manifest with machine-independent splits |
| `_audit_check.py` Gate C FAIL | SHA-256 duplicates across splits | Audit dataset for duplicate files |
| Shortcut audit FAIL | Model may exploit trivial features | Investigate metadata correlation, add regularization |
| Publication audit NO-GO | Critical gate failure | Address failing gates per their recommendations |
| Evaluate ValueError | Missing class in evaluation data | Ensure both normal and abnormal samples exist in the split |

---

*Status KEY: VERIFIED = executed with observed output; UNVERIFIED = code exists but not executed in this session; IMPLEMENTED = code ready but requires external resource (dataset/checkpoint).*
