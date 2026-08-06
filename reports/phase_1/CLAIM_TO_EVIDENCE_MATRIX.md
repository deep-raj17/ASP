# Claim-to-Evidence Matrix

| ID | Candidate claim | Current evidence | Missing evidence / minimum experiment | Eligibility now | Risk if claimed |
|---|---|---|---|---|---|
| 1 | CHAAD is a hybrid industrial acoustic anomaly-detection framework. | Hybrid modules import; checkpoint and training trace exist. | End-to-end final validation remains needed for performance, not identity. | Supported as a descriptive implementation claim. | Low if phrased descriptively. |
| 2 | The method supports machine-independent evaluation. | Manifest enforces train `id_04`, val `id_00/id_02`, test `id_06`; audit reports no overlap. | Exercise protected test only after full freeze. | Supported as protocol capability. | Low within audited scope. |
| 3 | Reliability-aware fusion improves anomaly detection. | `models/reliability.py` exists and imports. | Paired fixed/global/unconditioned/sample-dependent comparisons across seeds. | Not eligible. | Critical—the principal contribution would be unsupported. |
| 4 | Sample-dependent fusion outperforms fixed fusion. | No qualifying output. | Identical-backbone paired fusion ablation. | Not tested. | Critical. |
| 5 | Machine/noise conditioning improves performance. | Conditioning code is described; no controlled result. | Conditioned versus unconditioned ablation. | Not tested. | High. |
| 6 | Hybrid architecture outperforms individual branches. | No branch-isolation outputs. | CNN-only, temporal-only, AE-only, and hybrid matched runs. | Not tested. | High. |
| 7 | System generalises across machine IDs. | Validation machines differ from training machine, but ROC-AUC is only 0.6003. | Frozen final test on `id_06`, across seeds, with uncertainty. | Partially supported only as feasibility. | High if framed as strong generalisation. |
| 8 | Robust across noise conditions. | Stale/provisional subgroup tables; no final uncertainty. | Per-noise prespecified analysis across seeds and perturbation checks. | Not tested. | High. |
| 9 | Outperforms classical baselines. | Diagnostic RF/logistic results show a small advantage, but provenance and matched inputs are incomplete. | Publication-grade matched classical baselines and paired CIs. | Partially supported diagnostically, not publishable. | High. |
| 10 | Outperforms neural baselines. | Runner only. | Matched neural baselines under frozen protocol. | Not tested. | High. |
| 11 | Improvements stable across seeds. | One seed. | At least three prespecified independent seeds. | Not tested. | Critical. |
| 12 | Improvements statistically significant. | Analysis script only. | Paired predictions, effect sizes, CIs, tests, multiplicity policy. | Not tested. | Critical. |
| 13 | Model is well calibrated. | Calibration procedure exists; quality metrics do not. | Brier/ECE/reliability curve under a frozen calibration protocol. | Unsupported. | Medium/high. |
| 14 | Method is computationally practical. | No frozen latency/throughput/cost evidence. | Parameters, FLOPs, latency, throughput, memory on stated hardware. | Not tested. | Medium. |
| 15 | Experiment is reproducible. | Seed, manifest, checkpoint, logs, and partial config preserved. | Clean immutable source, full environment/command, and independent reruns. | Partially supported. | High if called fully reproducible. |
| 16 | Dataset is cryptographically equivalent to official release. | Eight archive MD5 matches, one mismatch, three missing; no fresh reference corpus. | Resolve PMPS-01 by verified acquisition/equivalence. | Contradicted. | Critical integrity risk. |
| 17 | No leakage detected in documented audit scope. | Machine isolation, cross-split hash audit, calibration, and threshold boundaries checked. | Repeat only if data/protocol changes. | Supported with scope qualifier. | Low if scope is explicit. |
| 18 | System is deployment ready. | Deployment code only. | Validated science, export parity, robustness, and target-hardware tests. | Unsupported. | High. |
| 19 | Method is a novel IEEE-level contribution. | Mechanism is implemented. | Current literature review plus contribution experiments and statistical evidence. | Unsupported. | Critical novelty/acceptance risk. |

## Freeze rule

Only claims 1, 2, and 17 are currently publication-eligible, and each must be
phrased narrowly. Claim 7, 9, and 15 may be described only as provisional
diagnostic findings or limitations. All other empirical contribution claims
must remain out of the manuscript until their minimum evidence is produced.
