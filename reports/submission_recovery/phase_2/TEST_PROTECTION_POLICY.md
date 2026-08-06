# Protected Test Policy

The test split is `id_06` with 12,747 manifest rows.

Before explicit Phase 8 authorization:

- no test row may be loaded by training, calibration, fusion fitting, model
  selection, checkpoint selection, threshold selection, debugging, pilots,
  baseline execution, ablation execution, or robustness analysis;
- no existing test-named prediction file may be opened for scientific use;
- development commands must declare `phase` and `split`;
- Phase 3 must add an executable guard that rejects `test` for phases 1–7;
- prediction exports must record split, manifest checksum, sample IDs,
  checkpoint hash, configuration hash, and phase;
- validation conclusions cannot cite test metrics.

Phase 8 requires a frozen authorization package and explicit user permission.
After test access, no model, fusion rule, checkpoint, threshold, claim-selection
policy, or statistical method may change.
