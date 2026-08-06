# Compute Budget

Host: NVIDIA GeForce RTX 4070 SUPER, CUDA 12.1.

The historical 100-epoch EfficientNet-B4 run took roughly 29 hours. The frozen
30-epoch cap therefore budgets approximately 9 GPU-hours per backbone run
before early stopping.

## Planned budget

| Work | Runs | Approx. GPU hours/run | Approx. total |
|---|---:|---:|---:|
| Phase 4 pilot: full backbone | 1 | 1–2 (bounded 4 epochs) | 2 |
| Phase 4 pilot: simple neural baseline | 1 | <1 | 1 |
| Full CHAAD backbone, three seeds | 3 | 9 | 27 |
| Simple neural baseline, three seeds | 3 | 3 | 9 |
| No-AE training ablation, three seeds | 3 | 9 | 27 |
| No-contrastive training ablation, three seeds | 3 | 9 | 27 |
| Fusion fits and analyses | 27+ posthoc fits | <0.2 | 6 |
| Contingency for one technical retry | 1 | 9 | 9 |
| Total ceiling | — | — | approximately 108 GPU-hours |

Storage is estimated at 0.3 GB per retained selected checkpoint plus logs and
predictions. Intermediate epoch checkpoints must not overwrite historical
files and should be written under unique run directories. No cleanup or
deletion is authorized.

## Feasibility gate

Phase 4 updates measured runtime and memory. If the projected ceiling exceeds
available compute, stop before Phase 5 and request a precise budget decision;
do not silently reduce seeds, methods, epochs, or evidence requirements.
