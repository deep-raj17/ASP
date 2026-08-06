# Reuse or Regenerate Decision

| Asset | Decision | Boundary |
|---|---|---|
| Manifest and split mapping | REUSE | disclose unresolved archive lineage |
| Selected checkpoint and epoch series | ARCHIVE / DIAGNOSTIC REUSE | not a final paper model |
| Corrected validation predictions | REUSE | provisional validation evidence only |
| Original validation predictions | EXCLUDE | duplicate identity corruption |
| `reports/test_predictions.csv` | QUARANTINE | misleading name; do not use |
| Corrected validation metrics | REUSE | independently recomputed |
| Training logs and curves | DIAGNOSTIC REUSE | document underfitting |
| Diagnostic baseline results | ARCHIVE | insufficient frozen protocol/provenance |
| Baseline runner | VALIDATE THEN REUSE | execute only after protocol freeze |
| Reliability-aware module | VALIDATE THEN REUSE | needs tests and controlled ablations |
| Statistical script | VALIDATE THEN REUSE | needs prespecified paired inputs |
| Calibration | REGENERATE OR RESTORE | only under explicit authorized workflow |
| Final baselines, ablations, tables, figures | REGENERATE | derive from registered predictions |
| Environment capture | REGENERATE PER RUN | current shared environment conflicts |
| Manuscript claims | REGENERATE | bind every claim to final evidence |
| Governance records | REUSE AS CONTEXT | never substitute for empirical evidence |
