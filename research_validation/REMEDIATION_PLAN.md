# Evidence Remediation Plan

This plan reconciles the attached four-status analysis with repository
evidence. It is an execution plan, not a claim that the work is complete.

## 1. PMPS-01 / scientific readiness — BLOCKED

Already verified locally: machine-independent manifest splits, duplicate-hash
checks, preprocessing leakage checks, corrected validation export, and the
53,046-file corpus audit. The unresolved PMPS blocker is authoritative dataset
release identity/provenance. Additional scientific evidence still required
before publication claims includes multi-seed stability, fair baselines,
controlled ablations, inferential statistics, and error analysis.

Required sequence:

1. Resolve dataset archive identity and licensing evidence.
2. Re-evaluate PMPS-01 without changing raw data or the test lock.
3. Only after authorization, run registered multi-seed, baseline, ablation,
   statistical, and error-analysis protocols.
4. Recompute the evidence and claim registries.

## 2. ROS master workflow — BLOCKED BY DESIGN

No ROS repair is required. ROS correctly stops downstream stages while PMPS-01
is blocked. Re-evaluate the workflow after the prerequisite evidence changes.

## 3. RMS workflow — NOT STARTED

RMS is post-submission lifecycle management. There is no submission receipt,
review record, acceptance, DOI, or publication event in the repository. Do not
create RMS records until an actual external event exists.

## 4. IEEE submission readiness — NOT READY

The repository has publication infrastructure but not a defensible submission
package. Manuscript drafting, venue compliance, and final packaging remain
blocked until the scientific evidence gates pass.

## Authorization boundary

This plan does not authorize retraining, test-set evaluation, package
installation, dataset downloads, or external submission.
