# Scientific Gap Analysis

## Mandatory publication blockers

1. **Dataset lineage and research freeze.** The local corpus is internally
   sound but official archive lineage is conflicting. Until a documented
   re-entry condition resolves PMPS-01, new scientific experiments are not
   authorized.
2. **Unsupported principal contribution.** Reliability-aware fusion is
   implemented but has no matched fusion ablation, no multi-seed effect size,
   and no statistical evidence.
3. **Weak provisional candidate.** The corrected selected-checkpoint validation
   ROC-AUC is 0.6003 and the training audit classifies the run as underfitting.
4. **Missing publication baselines.** Only a weakly reconstructed diagnostic
   subset exists. There are no matched classical and neural baseline packages.
5. **Missing ablations.** No experiment isolates reliability weighting,
   conditioning, individual branches, or fusion alternatives.
6. **Missing seed stability and statistics.** One seed cannot establish
   robustness, variance, effect size, confidence intervals, or significance.
7. **No protected-test result.** This is correct at Phase 1, but an IEEE paper
   ultimately needs a single-use final evaluation after every choice is frozen.
8. **Incomplete reproducibility package.** The historical dirty patch, exact
   command, full run config, and environment lock are absent.
9. **No evidence-bound IEEE manuscript.** Governance reports are extensive, but
   a scientific manuscript supported by completed experiments does not exist.

## High-value improvements

After mandatory evidence exists, add calibration quality, per-machine and
per-noise uncertainty, an error taxonomy, perturbation robustness, and
computational-cost measurements. These materially improve reviewer confidence,
but they cannot substitute for contribution ablations, baselines, multiple
seeds, or the protected final evaluation.

## Optional work

Edge deployment, a user interface, dashboards, commercialization, and extended
governance automation are outside the shortest submission path. They should be
omitted unless the paper makes a deployment claim.

## Minimum defensible evidence package

- resolved dataset governance and immutable experiment protocol;
- matched candidate, fixed-fusion, unconditioned, branch, classical, and neural
  baselines;
- at least three predefined seeds;
- identity-preserving validation predictions for every paired method;
- effect sizes, confidence intervals, and prespecified paired tests;
- one final frozen-threshold evaluation on protected `id_06`;
- complete provenance for every run;
- tables/figures regenerated exclusively from registered predictions;
- a manuscript whose claims map one-to-one to those artifacts.

Anything less requires reducing the central claim; it cannot be repaired with
additional governance prose.
