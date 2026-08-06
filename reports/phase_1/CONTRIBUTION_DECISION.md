# Publication Contribution Decision

## Verdict

**CONTRIBUTION PROMISING BUT UNVALIDATED**

## Assessment

| Question | Finding |
|---|---|
| Is it implemented? | Yes. `models/reliability.py` imports successfully and defines condition-aware, sample-dependent signal fusion. |
| Is it technically coherent? | Provisionally yes at code-structure level; no numerical or behavioral unit-test package proves all intended invariants. |
| Is it isolated? | Sufficiently modular to support controlled fusion ablations, but integration behavior must be frozen in Phase 2. |
| Experimentally tested? | No qualifying contribution experiment was found. |
| Compared with fixed fusion? | No. |
| Tested across seeds? | No; only one candidate run exists. |
| Meaningful effect size? | Unknown. |
| Statistical significance? | Unknown. |
| Generalised across machines? | The overall architecture was validated on unseen validation IDs with weak ROC-AUC; the reliability module's independent effect is unknown. |
| Generalised across noise? | Unknown. |
| Differentiated from prior work? | Not established by a current systematic literature adjudication. |
| Can it currently be the main paper contribution? | No. It may remain the proposed contribution, but cannot yet be claimed as effective or novel. |

## Minimum validation

Freeze a paired experiment that changes only the fusion mechanism:
fixed expert weights, equal weights, global learned weights, sample-dependent
unconditioned weights, and full machine/noise-conditioned reliability weights.
Run the same predefined seeds and preprocessing, retain sample-aligned
predictions, report effect sizes and uncertainty, and test the protected set
only after selection. A literature comparison must separately establish
differentiation; performance alone does not prove novelty.
