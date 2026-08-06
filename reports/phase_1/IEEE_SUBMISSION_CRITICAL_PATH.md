# IEEE Submission Critical Path

The repository is currently frozen by an external provenance dependency. The
following is the shortest defensible path after an authorized re-entry
condition occurs. It is a roadmap only; Phase 1 did not execute it.

| Order | Future phase | Objective | Inputs | Required outputs | Exit / failure criteria | Parallelism | Compute |
|---:|---|---|---|---|---|---|---|
| 0 | Governance re-entry | Resolve official dataset lineage and unfreeze research | Official archives or equivalent authorized evidence; current PMPS records | Verified acquisition/equivalence evidence; PMPS-01 reassessment | Exit only if governance authorizes continuation; otherwise remain frozen | No | Network/storage intensive |
| 1 | PHASE 2 — EXPERIMENTAL PROTOCOL FREEZE | Freeze hypotheses, variants, seeds, splits, thresholds, metrics, exclusions, compute budget, artifact IDs, and failure rules before runs | Phase 1 package; resolved dataset evidence | Signed/frozen protocol and immutable run matrix | Exit when every analysis choice is prespecified and executable; fail if contribution cannot be isolated | Some literature/protocol checks parallel | Low |
| 2 | Pipeline qualification | Verify tiny-batch overfit, identity-safe exports, deterministic mapping, and leakage guards without touching test | Frozen code/config; validation data | Qualification report and approved candidate configurations | Exit when training/evaluation is technically sound; fail/repair without test access | Baseline plumbing can parallel | Moderate |
| 3 | Candidate, baseline, ablation multi-seed runs | Produce evidence for contribution and performance | Qualified pipeline; frozen matrix | Registered checkpoints, logs, validation predictions for every method/seed | Exit when all prespecified runs complete; fail if effect is absent or unstable and reduce/redefine claim | Independent seeds/methods parallel | Very high |
| 4 | Statistical and robustness analysis | Quantify effect size, uncertainty, multiplicity, calibration, and subgroups | Sample-aligned frozen validation predictions | CIs, paired tests, robustness/calibration tables, decision memo | Exit if principal claim meets prespecified practical/statistical criteria; otherwise report negative result/redefine | Analyses parallel after inputs complete | Low/moderate |
| 5 | Final selection and protected test | Lock candidate, threshold, and analysis; perform one final `id_06` evaluation | Selection memo and final artifacts | Locked test predictions and final metric report | Exit on clean single-use evaluation; fail on integrity violation—do not tune on test | No | Moderate |
| 6 | Evidence-bound manuscript | Generate paper, figures, tables, limitations, and reproducibility package | Registered final evidence | IEEE draft, supplement, artifact manifest, claim audit | Exit when every empirical claim traces to evidence and independent QA passes | Writing/figure work partly parallel | Low |

No architecture tinkering, optional deployment work, dashboard work, or broad
governance expansion belongs on this critical path.
