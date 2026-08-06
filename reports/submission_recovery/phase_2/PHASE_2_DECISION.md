# Phase 2 Decision

Decision: **PASS**

## Frozen decisions

- One primary question and directional hypothesis.
- ROC-AUC primary metric; five secondary metrics.
- Seeds 42, 123, and 2026.
- Manifest-driven machine-independent splits.
- No protected test access before Phase 8.
- Thirty-epoch maximum with prespecified early stopping.
- Validation-loss checkpoint selection.
- Train-normal calibration.
- Nested out-of-fold validation fitting for learned fusion.
- Nine method rows and ten ablation rows.
- Paired seed-level and stratified prediction-level uncertainty analyses.
- Holm correction and explicit failed-run accounting.

## Calculable programme

The maximum backbone-training programme is 14 runs: one pilot full model, one
pilot simple neural model, then three seeds each for full CHAAD, simple neural,
no-autoencoder, and no-contrastive configurations. Fusion and branch
comparisons reuse registered backbone outputs. The estimated ceiling is about
108 GPU-hours including one technical retry.

## Exit criteria

| Criterion | Result |
|---|---|
| Same split/evaluation policy | PASS |
| Decisions frozen before new results | PASS |
| Test-derived decisions prohibited | PASS |
| Planned run count calculable | PASS |
| Outputs and failed-run handling specified | PASS |
| Claims map to experiments | PASS |
| Feasible minimum defined with compute gate | PASS |

## Next phase

Proceed to **PHASE 3 — EXPERIMENT INFRASTRUCTURE REPAIR**. No training or
protected test access occurred in Phase 2.
