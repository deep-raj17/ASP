# Phase 1 Quality Assurance Report

## Verdict

**PASS WITH DISCLOSED AUDIT INCIDENT**

The Phase 1 package is internally consistent and evidence-bounded. The
stateful `train.py --help` incident did not change a checkpoint or complete any
training; its temporary provenance overwrite was restored byte-for-byte and
the event stub was preserved.

## Automated structural checks

| Check | Result |
|---|---|
| Required outputs checked before QA report creation | 19 |
| Missing pre-QA outputs | 0 |
| Final required outputs after this report | 20 |
| CSV files parsed | 5 |
| CSV parse failures | 0 |
| Asset inventory rows | 655 before registering this QA report |
| Final asset inventory rows | 656 |
| Invalid asset status labels | 0 |
| Asset hash/path failures | 0 |
| Claim rows | 20 |
| Duplicate claim IDs | 0 |
| Experiment rows | 6 |
| Duplicate experiment IDs | 0 |
| Python files syntax-parsed | 99 |
| Python syntax failures | 0 |
| Central module imports checked | 10 |
| Central import failures | 0 |

The QA report itself is registered into the asset inventory after its final
bytes are written; the inventory intentionally does not contain a hash of
itself.

## Evidence integrity checks

- `artifacts/experiment_provenance.json` restored SHA-256:
  `fac6fa40c123caa48e8ab33d56149ceb07de71bcf43db1d55e9c83d6315aa459`.
  This exactly matches the earlier independent migration inventory record.
- `checkpoints/best_model.pt` SHA-256:
  `7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`.
  The selected model was not overwritten.
- Corrected validation predictions: 28,254 rows, 28,254 unique IDs, binary
  labels, finite scores, `val` split only, and machine IDs `id_00`/`id_02`.
- No protected-test computation occurred. The legacy file named
  `reports/test_predictions.csv` was inspected only and was correctly
  classified as validation content.
- Current manifest and checksum sidecar agree; the unexplained history of the
  earlier mismatch is disclosed rather than erased.

## Consistency review

- Final status, contribution verdict, claim matrix, gap register, and critical
  path agree that contribution evidence, baselines, ablations, multi-seed
  analysis, statistics, and a protected final result are missing.
- The raw archive checksum CSV governs the current count: eight matches, one
  mismatch, three missing. Conflicting five-match narratives are explicitly
  marked stale.
- Governance documents are not treated as empirical evidence without a cited
  lower-level artifact.
- No completion is inferred from script or checkpoint existence.
- The current state is called fully *reconstructed*, not fully reproducible,
  scientifically validated, or publication-ready.

## Historical preservation and scope

No historical evidence was intentionally replaced. During a permitted CLI
capability check, `train.py` unexpectedly entered stateful startup because it
does not support `--help`; this generated no epoch and was stopped. The
historical provenance file was restored exactly, and the new 88-byte event stub
was retained as incident evidence rather than deleted.

No dataset file was written, no model was retrained, no protected test split
was evaluated, PMPS-01 was not reassessed, and Phase 2 was not executed.
