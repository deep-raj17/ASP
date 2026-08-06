# Phase 3 Test Report

Status: **PASS**

| Check | Result |
|---|---|
| Full discovered suite | 100 tests passed |
| Python syntax parse | 103 files, zero failures before final report files |
| Frozen protocol parse | PASS |
| Development phase test rejection | PASS for phases 1–7 |
| Phase 8 authorization validation | PASS |
| Config serialization/hash stability | PASS |
| Immutable run contract | PASS |
| Prediction identity/completeness/finiteness | PASS |
| Baseline/ablation matrix validity | PASS |
| Grouped global-fusion OOF | PASS |
| Grouped reliability-gate OOF | PASS |
| Restricted historical checkpoint load | PASS |
| Real checkpoint CUDA BF16 forward | PASS; every output finite |
| CUDA FP16 diagnostic | FAILED as expected; non-finite transformer outputs |
| Validation metric repeatability | exact repeat |
| Protected Phase 3 test command | rejected before data access |

The suite count increased from 73 to 100 because publication-critical contract,
test-protection, grouped-CV, and artifact tests were added.
