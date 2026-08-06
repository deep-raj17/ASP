# Frozen Experiment Protocol Draft

**Status: PROPOSED — NOT AUTHORIZED**

This draft is prepared for human review only. It must not be executed until
dataset provenance closes and the protocol receives explicit authorization.

## Proposed design

- Question: Does reliability-aware fusion improve machine-independent anomaly
  detection over registered baselines under a fixed preprocessing and split
  protocol?
- Primary metric: validation ROC-AUC, with test evaluation deferred to the
  approved final gate.
- Secondary metrics: PR-AUC, EER, balanced accuracy, precision, recall, F1,
  and subgroup results.
- Independent unit: manifest-relative sample path; grouping unit: machine ID.
- Split: frozen manifest (`id_04` train, `id_00`/`id_02` validation,
  `id_06` test).
- Seeds: five proposed seeds (`17, 29, 41, 53, 67`); no seed is authorized.
- Selection: validation loss under a predeclared early-stopping rule; no test
  access for selection.
- Statistics: paired bootstrap confidence intervals and predeclared multiple
  comparison handling.

Required baselines, ablations, failure handling, and artifact naming are listed
in the authorization matrix. Any change requires a recorded protocol deviation.
