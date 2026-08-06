# Roadmap — CHAAD Project

## Final Vision

A publication-quality research system for condition-aware industrial acoustic anomaly detection, with a clearly defined novel contribution (reliability-aware fusion), rigorous experimental validation, and a conference-ready manuscript.

## Current Milestone

**Research Validation & Publication Preparation** — Score ≥ 90% on publication go/no-go audit.

## Roadmap Notation

- [x] Verified completed
- [ ] Planned (not started)
- [~] In progress
- [!] Blocked

---

## Immediate (Next 2 Weeks)

### [ ] Train model with deterministic seeds [!]
- **Dependency**: MIMII dataset at configured path
- **Output**: `checkpoints/best_model.pt`
- **Validation**: Loss convergence, reasonable val metrics
- **Risk**: Dataset unavailable at `E:\MIMII`

### [ ] Run calibration on train_normal
- **Dependency**: Trained checkpoint
- **Output**: Calibration statistics (μ, σ per signal, reference embeddings)
- **Command**: `python calibrate.py`
- **Validation**: Calibration report shows reasonable statistics

### [ ] Evaluate on held-out test set
- **Dependency**: Calibration + checkpoint
- **Output**: `checkpoints/eval_report_test.json`
- **Command**: `python evaluate.py --split test`
- **Validation**: Metrics computed on never-touched test split

### [ ] Run baseline comparisons (Gate D) [!]
- **Dependency**: Trained checkpoint
- **Output**: `reports/baseline_comparison.json` (11 baselines)
- **Command**: `python scripts/run_baselines.py`
- **Validation**: 8+ baselines reported, proposed method vs. baselines compared

### [ ] Run statistical validation (Gate F) [!]
- **Dependency**: Predictions CSV from baseline/evaluation runs
- **Output**: `reports/statistical_validation_report.json`
- **Command**: `python scripts/statistical_validation.py --predictions ...`
- **Validation**: Bootstrap CIs, DeLong test, McNemar test, effect sizes

## Short-Term (Within 1 Month)

### [ ] Implement ablation study runner (Gate E) [!]
- **Dependency**: Baseline infrastructure, trained model
- **Output**: `reports/ablation_study_report.json`
- **Ablations to run** (from `docs/NOVELTY_AND_CONTRIBUTIONS.md` Section H):
  1. Fixed-weight fusion (no reliability gate)
  2. Equal-weight fusion (uniform averaging)
  3. Global learned weights (sample-independent)
  4. Condition-agnostic reliability (no metadata input)
  5. Full proposed method (condition-aware reliability fusion)
  6. Component removal (remove one score source at a time)
- **Validation**: Each ablation shows contribution of removed component

### [ ] Implement robustness analysis (Gate H) [!]
- **Dependency**: Model predictions on test set
- **Output**: `reports/robustness_report.json`
- **Analyses**:
  1. False positive / false negative examination
  2. Score distribution per subgroup
  3. Calibration error (ECE, MCE)
  4. Gaussian noise perturbation at test time
  5. Audio perturbation (time stretch, pitch shift, volume)
- **Validation**: Failure modes characterized, limitations documented

### [ ] Generate publication manuscript (Gate J)
- **Dependency**: All evaluation results complete
- **Output**: `reports/manuscript.md` with LaTeX-ready tables and figures
- **Contents**: Introduction, related work, method, experiments, results, discussion, conclusion
- **Validation**: IEEE/ACM formatted, all claims supported by evidence

### [ ] Achieve publication audit GO verdict
- **Target**: ≥ 90% score on `scripts/run_publication_audit.py`
- **Dependency**: Gates D, E, F, H, J passing

## Medium-Term (1-3 Months)

### [ ] Multi-seed evaluation
- Run training with 5+ different seeds
- Report mean ± std for all metrics
- **Output**: Multi-seed stability analysis in manuscript

### [ ] Camera-ready formatting
- Format for target venue (IEEE/CVPR/ICASSP)
- Generate publication-quality figures (300+ DPI)
- Verify bibliography
- **Output**: Camera-ready PDF

### [ ] Artifact evaluation package
- Freeze exact dependencies with hashes
- Containerize with Docker
- Create reproducibility script
- **Output**: GitHub release with DOI

### [ ] Submit to conference
- Upload manuscript
- Submit supplementary material
- Open-source code release

## Long-Term (3+ Months)

### [ ] Edge deployment validation
- Deploy to Raspberry Pi
- Measure latency and throughput
- Compare edge vs. server performance
- **Output**: Edge deployment report

### [ ] Cross-dataset generalization
- Test on DCASE challenge data
- Test on ToyADMOS dataset
- **Output**: Generalization study

## Deferred Work (Needs Owner Approval)

- Integration with real-time streaming audio pipeline
- GUI dashboard for factory operators
- Multi-sensor fusion (vibration + acoustic)
- Continuous learning / online adaptation
- Federated learning across factory sites
- Integration with predictive maintenance systems

## Explicitly Out-of-Scope

- Unsupervised anomaly detection (current method is supervised)
- Real-time video anomaly detection
- Non-acoustic sensor data (vibration, temperature)
- Cloud-based deployment (current target is edge/on-premise)
- Multi-language documentation

---

*Last updated: 2026-07-21*
