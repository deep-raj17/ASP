# Test-Set Protection Record

**Status: ACTIVE**

The `id_06` test split is protected. It was not used for model selection,
threshold selection, architecture decisions, ablations, or informal debugging
in this remediation phase. The proposed protocol keeps all new development on
train/validation data and defers any test evaluation until leakage, split,
provenance, and authorization gates pass.
