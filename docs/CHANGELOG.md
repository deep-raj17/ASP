# Changelog

## 2026-07-27

### Fail-closed production inference

- Added a shared artifact requirement for calibrated production inference.
- Prevented the Gradio UI and Flask API from serving random-weight or
  uncalibrated scores when required artifacts are absent.
- Switched model and calibration loading to restricted PyTorch deserialization.
- Added three regression tests; the full suite passed with 73 tests.
- Verified the current runtime is BLOCKED by the missing detector calibration.

## 2026-07-24

### ROS core, CLI, and CHAAD adapter

- Added and tagged ROS workflow, evidence, registry, and CLI milestones.
- Added the PMPS 1.0.0 machine-readable workflow and CHAAD adapter.
- Imported 18 evidence records and ten registry records append-only.
- Preserved the invalid and corrected validation exports separately.
- Derived PMPS-01 as BLOCKED through the evidence and workflow engines.
- Completed a read-only 53,046-file/135.8 GB corpus audit with zero errors.
- Kept authoritative local dataset-release identity as the sole PMPS-01
  blocker; no publication-readiness claim was made.

### PMPS-01 provenance snapshot

- Generated an isolated PMPS-01 package under
  `artifacts/publication_baseline/` without retraining, checkpoint changes,
  architecture changes, or test-set evaluation.
- Inventoried and SHA-256 hashed all 101 checkpoint files.
- Captured repository, environment, configuration, Git, dataset, checkpoint,
  and experiment provenance plus model/dataset cards and research governance.
- Verified all 53,046 manifest paths exist, recorded file sizes match, and the
  current manifest SHA-256 matches its sidecar.
- Marked PMPS-01 **FAIL** because only one representative WAV was live-decoded;
  full-corpus decode, NaN/Inf scanning, and live hash recomputation remain
  unverified. PMPS-02A through PMPS-03C were not executed.
- Corrected stale documentation that claimed the checkpoint was absent and the
  current manifest sidecar mismatched.

### Validation prediction export fix

- Fixed the EXP-CHAAD-001 validation prediction exporter so `sample_id` comes
  from the dataset-level normalized manifest `relative_path`, not
  `batch_idx * len(labels)`.
- Regenerated validation predictions from `checkpoints/best_model.pt` without
  retraining, architecture changes, split changes, checkpoint overwrite, or
  test-set evaluation.
- Saved corrected artifacts:
  `validation_predictions_corrected.csv`,
  `prediction_export_validation.json`,
  `prediction_export_determinism.json`,
  `independent_metrics_corrected.json`, and
  `subgroup_metrics_corrected.json`.
- Verified corrected export: 28,254 expected validation samples, 28,254 rows,
  28,254 unique sample IDs, 0 duplicates, 0 missing IDs, 0 invalid labels, 0
  non-finite scores, and validation-only split membership.
- Verified batch-size determinism for batch sizes 16 and 32 after disabling TF32
  for deterministic inference: max score difference 2.384185791015625e-07,
  ROC-AUC difference 0.0.
- Corrected validation metrics: ROC-AUC 0.6002609445, PR-AUC 0.2578861055,
  EER 0.4264914172, Youden threshold 0.4995152950.

### Ordered diagnostic audit correction

- Independently checked `validation_predictions.csv` and found 30 duplicated
  `sample_id` values (60 rows), caused by the batch-local ID formula across a
  short final batch.
- Reclassified the evaluation audit as **BUG CONFIRMED**; no production code or
  checkpoint was changed.
- Reclassified the training audit as **MODEL UNDERFITTING** and generated the
  four required TensorBoard-derived training curves.

### Experiment preservation

- Verified `checkpoints/best_model.pt` exists and matches the copied
  `artifacts/EXP-CHAAD-001/checkpoint.pt` by SHA-256.
- Completed the EXP-CHAAD-001 preservation metadata with the selected epoch,
  checkpoint criterion, exact checkpoint/final metrics, environment evidence,
  and unresolved provenance.
- Registered EXP-CHAAD-001 as an **OFFICIAL PROVISIONAL RESULT**.
- Updated the current state to reflect that a preserved checkpoint now exists.
- Detected and documented (without concealing or rewriting) a mismatch between
  the current dataset manifest and `metadata/dataset_manifest.sha256`; the
  experiment package records the calculated hash.
- No training, architecture modification, test-set evaluation, commit, or push
  was performed.
### 2026-07-24 — ROS-PUB-01 prerequisite gate

- Evaluated ROS-PUB-01 against the repository's machine-readable ROS state.
- Recorded a deterministic **BLOCKED** decision because ROS-PROJECT-02 through
  ROS-PROJECT-13 are missing and PMPS-01 remains blocked on dataset provenance.
- Deliberately did not generate novelty, literature, falsification, or
  manuscript-readiness artifacts and did not start ROS-PUB-02.

### 2026-07-24 — ROS-PUB-02/03 prerequisite gates

- Recorded ROS-PUB-02 and ROS-PUB-03 as **BLOCKED** because their required
  predecessor stages did not pass.
- Did not generate venue recommendations or manuscript architecture artifacts.

### 2026-07-24 — ROS-PUB-04 prerequisite gate

- Recorded ROS-PUB-04 as **BLOCKED** because ROS-PUB-03 did not pass.
- Did not generate independent audit, adversarial review, or submission
  certification artifacts.

### 2026-07-24 — ROS-PUB-05 prerequisite gate

- Recorded ROS-PUB-05 as **BLOCKED** because ROS-PUB-01 through ROS-PUB-04 did
  not pass.
- Did not assemble a venue submission package or claim compliance.

### 2026-07-24 — ROS-DEPLOY-01 / ROS-SEC-01 prerequisite gates

- Recorded deployment and security subsystem gates as **BLOCKED** by incomplete
  upstream ROS project/publication domains.
- Did not make deployment-readiness or security-assurance claims.

- ROS-SEC-01 Parts 2 and 3 were checked and remained blocked; no partial
  security assessment was recorded as complete.

### 2026-07-24 — ROS-SEC-01 final and ROS-DATA-01 prerequisite gates

- Recorded ROS-SEC-01 Part 4 as blocked and ROS-DATA-01 as **BLOCKED** by
  incomplete upstream domains.
- Did not make data-governance, FAIR, provenance, or lifecycle claims.

### 2026-07-24 — ROS extension architecture proposal

- Recorded the proposed ROS-ML, RV, RW, RR, and RP hierarchy and recommended
  implementation order.
- Kept all proposed frameworks explicitly planned; no stage was marked complete.

### 2026-07-24 — ROS-IEEE-MASTER-01 Phase 0 inventory

- Created the scientific repository inventory, artifact registry, provenance
  registry, and missing-evidence register.
- Stopped the master pipeline at **BLOCKED** before experiments or manuscript
  drafting because prerequisite evidence is incomplete.

### 2026-07-24 — PMPS-01 requirement reconstruction

- Reconstructed all 13 PMPS-01 requirements from the machine-readable workflow.
- Confirmed 12 requirements with verified evidence and one blocking
  `dataset_license_identity` requirement.
- Stopped before training, test evaluation, and further validation phases.

### 2026-07-24 — Provenance closure and protocol draft

- Added dataset provenance matrix, hash registry, license record, and
  certification; local archive identity remains unavailable.
- Reconciled existing leakage/split evidence without repeating expensive scans.
- Drafted a frozen experiment protocol and authorization matrix with every row
  explicitly `PROPOSED`; no experiment was authorized or executed.

### 2026-07-24 — PMPS-01 license-identity closure

- Added license matrix, source-traceability record, citation record, and a
  certification that keeps PMPS-01 **BLOCKED**.
- Reduced the remaining blocker to exactly one action: recover authoritative
  archive identity evidence for the local corpus.

- Verified from the official Zenodo record that twelve archive MD5 checksums
  are published; local comparison remains unperformed because the original
  archive/acquisition record is unavailable.

### 2026-07-24 — PMPS-01 traceability adjudication

- Classified the blocker as **Category A**: required official identity evidence
  exists, but local archive linkage is missing.
- Retained PMPS-01 **BLOCKED** and documented one recovery action.

### 2026-07-24 — Local MIMII archive identity verification

- Found nine official-named local ZIPs and computed their MD5s read-only.
- Five matched official values; `0_dB_valve.zip` mismatched; three official
  archives were missing.
- Reclassified PMPS-01 provenance from missing evidence to **CONFLICTING**.

### 2026-07-27 — Local MIMII root forensic validation

- Reconciled the canonical `E:\MIMII` root against the manifest and prior
  full-corpus audit: 53,046 extracted files are present and internally
  consistent.
- Documented that ZIP-container discrepancies remain unresolved and do not by
  themselves establish extracted-audio corruption.

### 2026-07-27 — Fresh acquisition preflight stop

- Created isolated `E:\MIMII_VERIFIED_ACQUISITION` and captured official Zenodo
  manifest metadata.
- Stopped before download because 72.5 GB free space was insufficient for the
  100.2 GB archive set plus extraction overhead.
- Preserved the historical dataset and did not start PMPS-02.

### 2026-07-27 — Acquisition storage migration

- Retired the empty E: acquisition location without deleting it.
- Created isolated `C:\MIMII_VERIFIED_ACQUISITION` and validated NTFS capacity,
  isolation, and write access.
- Marked acquisition storage **READY**; no downloads or extraction started.

### 2026-07-27 — COAP-01 acquisition stop

- Started the guarded sequential downloader in the isolated C: root.
- The first Zenodo transfer stalled; two partial attempts were preserved and
  the protocol stopped before checksum acceptance or extraction.
- Historical data and scientific artifacts were not modified.

### 2026-07-27 — NAFR-01 network forensics

- Captured Windows/network/server diagnostics and transfer timeline.
- Classified the acquisition failure as **ROOT CAUSE LIKELY IDENTIFIED**:
  remote server limitation, medium confidence.
- No additional download, extraction, or dataset access occurred.

### 2026-07-27 — RPGS-01 governance decision

- Assessed collected provenance evidence and research risks without new
  acquisition or dataset access.
- Recommended pausing research progression pending verified official acquisition.

### 2026-07-27 — RDRP-01 research freeze

- Froze the current research state under an external acquisition dependency.
- Registered 99 generated artifacts with SHA-256 metadata and documented
  dependencies, re-entry conditions, and dormancy certification.

### 2026-07-27 — IRRA-01 readiness assessment

- Completed an independent evidence-only IEEE readiness review.
- Verdict: **NOT READY FOR IEEE SUBMISSION**; August submission: **NO**.

### 2026-07-24 — Evidence remediation plan

- Reconciled the four blocked/not-started statuses into an evidence-first
  remediation sequence.
- Distinguished already verified leakage/split checks from unresolved dataset
  provenance and unexecuted publication evidence.

- ROS-DATA-01 Part 2 was checked and remained blocked; no partial data
  assessment was recorded as complete.

- ROS-DATA-01 Part 3 was checked and remained blocked by upstream prerequisites.

### 2026-07-24 — ROS-DATA-01 final gate

- Recorded Part 4 as blocked; no end-to-end data-governance certificate was
  issued.

### 2026-07-27 — Phase 1 scientific asset and claim–evidence freeze

- Captured the dirty repository, environment, complete file-level SHA-256
  inventory, code capability map, experiment registry, and result
  recomputation audit under `reports/phase_1/`.
- Recomputed the corrected EXP-CHAAD-001 validation metrics without training or
  protected-test access and froze 20 candidate claims against available
  evidence.
- Classified the scientific state as fully reconstructed, publication as
  blocked by missing or corrupted assets, and the reliability-aware
  contribution as promising but unvalidated.
- Recorded that the raw archive comparison contains eight MD5 matches, one
  mismatch, and three missing containers, superseding stale five-match
  narratives for inventory purposes without changing PMPS-01.
- A `train.py --help` capability probe exposed a stateful CLI startup. No epoch
  completed and no checkpoint changed; the historical run-start provenance was
  restored byte-for-byte to its registered SHA-256, and the 88-byte event stub
  was retained as non-scientific incident evidence.
- Phase 2 was not executed.

### 2026-07-27 — Local planner-to-Codex orchestrator

- Added a resumable local controller linking an OpenAI Responses API planner
  to sandboxed, non-interactive `codex exec` tasks.
- Added master-goal, ignored state/runtime, setup documentation, an initial
  completion report, and focused offline tests.
- Added dirty-worktree, credential-scrubbing, planner-schema, human-risk,
  research-re-entry, protected-test, timeout, iteration, and report-completion
  gates.
- Verified syntax, 11 focused tests, controller preflight, and dry-run mode
  without an API request, nested Codex task, training, or protected-test access.
