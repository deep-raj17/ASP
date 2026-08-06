# Known Issues — CHAAD Project

> Last updated: 2026-07-21

## Confirmed Issues

### ISSUE-001: No independent test-set evaluation exists
- **Severity**: HIGH
- **Status**: UNVERIFIED
- **Description**: The only available evaluation report (`checkpoints/eval_report.json`) was computed on the validation set, not the held-out test set. The reported ROC-AUC of 99.99997% cannot be treated as a final result.
- **Reproduction**: `checkpoints/eval_report.json` shows validation split evaluation only.
- **Expected**: `python evaluate.py --split test` should produce independent metrics.
- **Workaround**: Run evaluation on test split with a frozen validation threshold.
- **Fix**: Execute: `python evaluate.py --split test`

### ISSUE-002: PMPS-01 full-corpus integrity verification incomplete
- **Severity**: HIGH (BLOCKER)
- **Status**: RESOLVED FOR CURRENT FILES; RELEASE IDENTITY STILL BLOCKED
- **Description**: The historical PMPS-01 report decoded one representative
  WAV. A new non-overwriting audit has now decoded all 53,046 current files,
  scanned them for non-finite values, checked metadata, and recomputed hashes.
- **Evidence**:
  `migration/chaad/remediation/pmps01_dataset_audit_20260724.json`.
- **Remaining**: The extracted local files are not tied cryptographically to
  the official public 1.0 ZIP checksums because the archives or download
  receipt are unavailable. PMPS-01 remains BLOCKED.

### ISSUE-003: Publication audit score below GO threshold
- **Severity**: MEDIUM
- **Status**: CONFIRMED
- **Description**: `scripts/run_publication_audit.py` returns 57.1% (CONDITIONAL). Gates D, E, F, H, J are NOT_CHECKED.
- **Fix**: Execute all remaining gates after training.

### ISSUE-004: scipy RuntimeWarning in shortcut audit
- **Severity**: LOW
- **Status**: CONFIRMED (harmless)
- **Description**: `scripts/audit_shortcuts.py` produces: `RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation` during duration t-test.
- **Cause**: Nearly identical duration values across normal/abnormal samples.
- **Fix**: None needed. Suppress warning or verify data is correctly uniform.

### ISSUE-005: Provenance file not recorded
- **Severity**: LOW
- **Status**: PARTIALLY RESOLVED
- **Description**: `artifacts/experiment_provenance.json` is from the
  EXP-CHAAD-001 run and captures the commit, interpreter, PyTorch/CUDA versions,
  seed, and selected configuration fields. It does not capture the exact shell
  command, complete config, dirty-tree patch, GPU identity, or dependency lock.
- **Fix**: Extend future run-start provenance before another official training
  run; do not retroactively infer the missing EXP-CHAAD-001 fields.

### ISSUE-006: Origin of manifest-sidecar correction is unknown
- **Severity**: MEDIUM
- **Status**: PARTIALLY RESOLVED
- **Description**: The current manifest and sidecar now match exactly at
  `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`.
  Earlier session evidence recorded a one-character mismatch. The current byte
  match is VERIFIED, but no committed history explains when or by whom the
  sidecar changed.
- **Evidence**: Independent PowerShell SHA-256 check and
  `artifacts/publication_baseline/dataset_inventory.json`.
- **Resolution needed**: Record formal approval/provenance for the authoritative
  checksum in version history.

## Suspected Issues (UNVERIFIED)

### ISSUE-S01: Old evaluation may have used stale config
- The `checkpoints/eval_report.json` (99.99997% ROC-AUC) was produced before the machine-independent split protocol was formalized. It may reflect a different split configuration.
- **Need**: Compare eval_report.json against the current manifest-derived splits.

### ISSUE-S02: Reported metrics may be at ceiling
- ROC-AUC 99.99997% and PR-AUC 99.99986% suggest near-perfect separation. This may indicate a data issue (leakage in old split) or genuine ease of the task.
- **Need**: Independent test-set evaluation to verify.

## Environment-Specific Issues

### ISSUE-E01: Windows path assumptions
- `config.py` hardcodes `E:\MIMII` with a Windows raw string. Cross-platform users must update manually.
- **Fix candidate**: Use environment variable `MIMII_DATASET_DIR`.

### ISSUE-E02: num_workers=0 required on Windows
- PyTorch DataLoader multiprocessing fails on Windows with `num_workers > 0`. Config enforces 0.
- **Note**: This is documented but may surprise Linux users.

## Resolved Issues

| Issue | Resolution | Date |
|-------|-----------|------|
| Stale `machine_dependent` split config | SUPERSEDED by manifest-based split | 2026-07 |
| `_inspect_manifest.py` temporary file | DELETED | 2026-07 |
| Label-type handling in audit scripts | Fixed: `_parse_labels()` handles both string and numeric | 2026-07-21 |
| UnicodeDecodeError in publication audit | Fixed: added `encoding="utf-8"` to file reads | 2026-07-21 |
