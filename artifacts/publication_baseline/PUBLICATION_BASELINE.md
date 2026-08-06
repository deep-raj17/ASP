# Publication Baseline — PMPS-01

## Status

**FAIL — PMPS-02 is not authorized by the PMPS-01 quality gate.**

## Verified

- Repository, environment, configuration, Git, checkpoint, and experiment inventories captured.
- Manifest checksum matches its sidecar: `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`.
- All 53,046 manifest paths exist and recorded sizes match.
- Existing repository validators pass split isolation and manifest duplicate checks.
- `best_model.pt` SHA-256 is recorded in `checkpoint_registry.json`.

## Unverified / incomplete

- Every WAV has not been freshly decoded in this stage.
- Every decoded tensor has not been scanned for NaN/Inf.
- Every current WAV hash has not been recomputed from the 135.8 GB corpus.
- Dataset version and redistribution license remain unknown.
- Cross-platform and fresh-environment reproduction remain unverified.

## Conflict resolution

`docs/CURRENT_STATE.md` and `docs/KNOWN_ISSUES.md` record a manifest-sidecar
mismatch, but the current files match exactly. Current file bytes and the
executed SHA-256 check supersede those stale statements.

## Publication risk

High until full-corpus integrity is executed and dataset identity/license are
resolved. Model evidence is validation-only and underfit.
