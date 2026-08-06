# Reuse / Regenerate Matrix

| Asset | Decision | Reason and publication boundary |
|---|---|---|
| `artifacts/EXP-CHAAD-001/checkpoint.pt` | REUSE AFTER VALIDATION | Preserve as diagnostic candidate and historical anchor; do not promote as final model without protocol-authorized comparison. |
| Corrected validation predictions | REUSE AS-IS | Identity, counts, split, finite scores, determinism, and metrics are validated; use only as provisional validation evidence. |
| Original validation predictions | ARCHIVE ONLY | Known duplicate-ID corruption. |
| `reports/test_predictions.csv` | EXCLUDE FROM PUBLICATION | Misleading filename and validation IDs; may remain for forensic comparison only. |
| Training logs and curves | REUSE AFTER VALIDATION | Useful to reconstruct underfitting; exact run provenance remains incomplete. |
| Corrected validation metrics | REUSE AS-IS | Recomputed from the validated corrected export; label them validation-only. |
| Legacy near-perfect metrics | EXCLUDE FROM PUBLICATION | Contradicted by lower-level identity-safe evidence/current protocol. |
| Dataset manifest | REUSE AS-IS | Current bytes/counts/splits validated; accompany with archive-lineage disclosure and unresolved correction history. |
| Split definitions | REUSE AS-IS | Machine-independent isolation validated within scope. |
| Baseline runner code | REUSE AFTER VALIDATION | Imports/help work; scientific outputs must be regenerated under the frozen protocol. |
| Diagnostic baseline output | ARCHIVE ONLY | Incomplete provenance and report/JSON inconsistencies prevent publication use. |
| Reliability-aware fusion implementation | REUSE AFTER VALIDATION | Technically coherent/importable; needs code tests and controlled contribution experiments. |
| Statistical-analysis script | REUSE AFTER VALIDATION | Capability exists; must be checked against frozen paired inputs and a prespecified analysis plan. |
| Existing training figures | ARCHIVE ONLY | Diagnostic reconstruction only; not final multi-seed paper figures. |
| Existing report tables | REGENERATE | Final tables must derive from registered, validated prediction exports. |
| Environment snapshot | REGENERATE | The current host differs from the historical training interpreter; freeze per future run. |
| Paper/manuscript material | REGENERATE | Existing narrative contains unsupported and stale claims. |
| Governance and provenance records | REUSE AS-IS | Preserve as audit context, but do not treat them as empirical scientific evidence. |
| Epoch checkpoints 1–100 | ARCHIVE ONLY | Retain selection history; do not count as independent runs. |
| Accidental Phase 1 TensorBoard stub | ARCHIVE ONLY | Startup side effect, no epoch/data result, not scientific evidence. |
