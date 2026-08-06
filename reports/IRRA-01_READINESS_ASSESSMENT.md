# IRRA-01 IEEE Research Readiness Assessment

**Assessment date:** 2026-07-27  
**Final verdict: NOT READY FOR IEEE SUBMISSION**  
**August readiness: NO**

## Executive summary

CHAAD is a hybrid industrial acoustic anomaly-detection system combining a
CNN/temporal model, autoencoder and contrastive signals with a proposed
reliability-aware fusion module. The repository has strong engineering,
integrity-audit, and evidence-preservation infrastructure. The preserved
experiment has a corrected validation export (28,254 unique samples) but only
modest validation discrimination (ROC-AUC 0.6002609445) and was classified as
underfitting.

The project is not publication-ready because baseline, multi-seed, ablation,
statistical, error-analysis, and robustness evidence is incomplete; dataset
archive provenance is conflicting; and no complete manuscript or independent
review package exists.

## Project and architecture

The pipeline is Dataset → manifest-based machine-independent split → audio
preprocessing → hybrid training → calibration → validation/test evaluation →
deployment. ROS/PMPS add workflow, registry, evidence, provenance, and gate
control around the research lifecycle. The architecture is modular and
traceable, but downstream ROS publication/deployment/security/data gates are
blocked by PMPS-01.

## Contribution assessment

The claimed contribution is the reliability-aware fusion module that learns
sample-dependent weights over multiple anomaly signals. Based on repository
evidence, this is a **moderate proposed methodological contribution**, not a
verified novel or superior scientific contribution. No completed ablation,
fair baseline comparison, multi-seed stability, or literature comparison
demonstrates incremental value or novelty.

## Implementation status

| Subsystem | Status | Approx. completion | Evidence/dependency |
|---|---|---:|---|
| Core model and training code | COMPLETED/UNVERIFIED | 80% | `train.py`, `models/`, preserved run |
| Evaluation/export pipeline | COMPLETED with historical bug correction | 90% | corrected export audit |
| Leakage and split audits | COMPLETED within scope | 90% | audit reports and manifest |
| Dataset provenance | BLOCKED | 75% | archive conflict and failed acquisition |
| Multi-seed experiments | NOT STARTED | 0% | authorization draft only |
| Baselines | NOT CERTIFIED | 20% | scripts exist; evidence incomplete |
| Ablations | NOT STARTED | 10% | plan only |
| Statistical validation | NOT STARTED | 20% | infrastructure exists |
| Error/robustness analysis | NOT STARTED | 15% | plans/docs only |
| Manuscript | NOT STARTED | 10% | no evidence-grounded draft |
| Independent review | NOT STARTED | 0% | blocked upstream |

## Experimental review

The preserved EXP-CHAAD-001 checkpoint and corrected validation predictions
provide a reproducible provisional result. Validation export integrity is
verified, but the result is weak and underfitting. Test-set evaluation was not
performed in the audit sequence. There are no certified fair baselines,
controlled ablations, multi-seed statistics, confidence intervals, effect
sizes, or comprehensive error/robustness analyses. The current evidence is
insufficient for an IEEE contribution claim.

## Dataset and reproducibility review

The extracted `E:\MIMII` corpus has 53,046 readable, finite, metadata-consistent
files with current SHA-256 matches and machine-independent splits. However,
five historical archive MD5s match, one differs, and three archive containers
are unavailable. COAP-01/02 could not create a verified reference corpus due
to network timeouts. Dataset identity is therefore internally plausible but
not cryptographically certified for the historical bytes.

Reproducibility rating: **Moderate**. Code, configuration, checkpoint, seed
utilities, manifests, and reports are preserved, but exact dataset acquisition
lineage and multi-run stability are unresolved.

## Software and governance review

Engineering maturity is strong: modular ROS components, append-only registries,
CLI tests, evidence checksums, and documented safeguards. Governance work was
meaningful where it produced machine-readable state and forensic evidence, but
the many blocked downstream prompt stages are redundant once the external
dependency is frozen. No further governance expansion is recommended during
dormancy.

## Publication scores (0–10)

| Category | Score | Basis |
|---|---:|---|
| Novelty | 4 | Proposed fusion idea; no comparative proof |
| Technical depth | 6 | Substantial hybrid architecture and implementation |
| Experimental quality | 3 | One underfitting provisional run |
| Statistical rigor | 2 | No certified inferential analysis |
| Writing quality | 2 | No complete manuscript |
| Figures/tables | 2 | No final evidence-grounded package |
| Related work | 2 | No audited literature comparison |
| Threats to validity | 5 | Limitations documented, experiments incomplete |
| Reproducibility | 5 | Strong artifacts, unresolved acquisition lineage |
| Dataset transparency | 4 | Detailed manifest, conflicting archive provenance |

## Remaining blockers

1. **Critical/external:** complete verified official acquisition or equivalent
   institutional evidence; current network transfer is blocked.
2. **Critical/internal:** multi-seed, baseline, ablation, statistical,
   error, and robustness evidence is absent.
3. **High/internal:** no evidence-grounded manuscript, figures, tables, or
   independent review package.
4. **High:** current validation discrimination is modest and underfitting was
   observed.

## August feasibility

**NO.** An August IEEE submission is not realistic from the frozen state. Before
submission, the project would need a resolved provenance decision, frozen and
executed experiments, statistically supported comparisons, contribution
validation, limitations, manuscript drafting, and independent review. The
required work is evidence-generating and cannot be compressed into a minor
editorial revision.

## Critical path

Resume only when official acquisition or equivalent evidence becomes available;
reassess PMPS-01; then execute the already-drafted authorized experiment plan;
validate baselines, ablations, statistics, errors, and robustness; and only
then draft and review an IEEE manuscript. This is a roadmap, not an authorization
to execute during dormancy.

## Independent verdict

**NOT READY FOR IEEE SUBMISSION.** The project demonstrates serious engineering
and integrity work, but the central scientific evidence package and complete
provenance chain are not yet defensible for peer review.
