# CHAAD Publication-Quality Research Pipeline - Master Task Tracker

## Phase 0: Audit Baseline ✅
- [x] Run _audit_check.py - all critical gates PASS
- [x] Verify machine-independent 3-split protocol
- [x] Confirm no data leakage (SHA, segments, machine IDs)
- [x] Verify calibration and threshold protocols
- [x] Document remaining gaps (Shortcut Learning ❓, Reproducibility ❓)

## Phase 1: Reproducibility Hardening
- [ ] Freeze exact dependencies (pip freeze → requirements_lock.txt with hashes)
- [ ] Add deterministic seed control across all random sources (torch, numpy, random, cuda)
- [ ] Generate environment_report.json with commit hash, Python version, CUDA version, hardware specs
- [ ] Add experiment_provenance.json with exact command line, git commit, timestamp
- [ ] Verify two independent runs produce identical results with same seed

## Phase 2: Shortcut Learning Audit
- [ ] Implement metadata-only baseline (predict anomaly from machine_type + noise_condition alone)
- [ ] Run feature permutation test: shuffle each metadata column, measure AUC drop
- [ ] Test if model exploits recording-level artifacts (train on 1 machine, test on another)
- [ ] Generate shortcut_learning_report.json

## Phase 3: Baseline Comparisons
- [ ] Implement Single-Score Baselines (reconstruction-only, embedding-only, mahalanobis-only, contrastive-only)
- [ ] Implement Equal-Weight Fusion baseline (uniform averaging of 4 calibrated scores)
- [ ] Implement Global Learned Weights baseline (one weight vector learned on validation)
- [ ] Implement Fixed-Weight Fusion baseline (current hand-selected weights from config)
- [ ] Implement Feature-based ML baselines (isolation forest, LOF, OCSVM on embeddings)
- [ ] Run all baselines on frozen test set, report with confidence intervals
- [ ] Generate baseline_comparison_report.json

## Phase 4: Novel Contribution - Reliability-Aware Fusion
- [ ] Design ReliabilityEstimator module (g_φ: embedding + metadata → reliability scores)
- [ ] Implement condition-aware reliability network in models/reliability.py
- [ ] Implement training script for reliability estimator (validation-stage, pairwise ranking loss)
- [ ] Add reliability-weighted score fusion to inference/detector.py
- [ ] Run full pipeline: train backbone → calibrate → train reliability gate → evaluate on test
- [ ] Generate reliability_fusion_results.json

## Phase 5: Ablation Studies
- [ ] Ablation 1: Fixed-weight fusion (no reliability gate)
- [ ] Ablation 2: Equal-weight fusion (uniform averaging)
- [ ] Ablation 3: Global learned weights (sample-independent)
- [ ] Ablation 4: Condition-agnostic reliability (no metadata input)
- [ ] Ablation 5: Full proposed method (condition-aware reliability fusion)
- [ ] Ablation 6: Component-removal study (remove one score source at a time)
- [ ] Generate ablation_study_report.json with significance tests

## Phase 6: Statistical Validation
- [ ] Implement bootstrap confidence intervals (1000 resamples) for all metrics
- [ ] Implement multi-seed evaluation (5+ seeds) with mean ± std reporting
- [ ] Add McNemar's test for paired comparison between methods
- [ ] Add Wilcoxon signed-rank test for score distributions
- [ ] Generate statistical_validation_report.json

## Phase 7: Subgroup Analysis
- [ ] Evaluate per machine_type (fan, pump, slider, valve)
- [ ] Evaluate per noise_condition (-6_dB, 0_dB, 6_dB) 
- [ ] Evaluate per machine_id cross-generalization
- [ ] Evaluate per label subgroup (normal vs abnormal score distributions)
- [ ] Generate subgroup_analysis_report.json

## Phase 8: Robustness & Failure Analysis
- [ ] Confusion analysis: examine false positives and false negatives
- [ ] Score distribution analysis per subgroup
- [ ] Calibration error analysis (ECE, MCE) per condition
- [ ] Adversarial robustness check (Gaussian noise injection at test time)
- [ ] Audio perturbation study (time stretch, pitch shift, volume change)
- [ ] Generate robustness_report.json

## Phase 9: Publication-Quality Evaluation Report
- [ ] Create comprehensive evaluation_report.md with all results
- [ ] Generate LaTeX-quality tables (baseline comparison, ablation, subgroup)
- [ ] Generate publication-quality figures (ROC curves, PR curves, calibration plots)
- [ ] Create consolidated metrics dashboard
- [ ] Generate final_evaluation_report.json

## Phase 10: Go/No-Go Audit Framework
- [ ] Create independent audit script (run_publication_audit.py)
- [ ] Implement all decision gates with pass/fail criteria
- [ ] Add automatic report generation
- [ ] Create go_no_go_checklist.md
- [ ] Run final audit and generate verdict

## Phase 11: Manuscript Generation
- [ ] Generate IEEE/ACM formatted manuscript draft
- [ ] Auto-populate tables and figures from evaluation results
- [ ] Generate bibliography
- [ ] Create supplementary material package
- [ ] Generate reproducibility package (code snapshot, configs, checkpoints)

## Phase 12: Camera-Ready & Submission Package
- [ ] Format for target venue (IEEE/CVPR/ICASSP template)
- [ ] Verify all figures are 300+ DPI, vector where possible
- [ ] Create GitHub release with all artifacts
- [ ] Prepare artifact evaluation package
- [ ] Generate final submission checklist
