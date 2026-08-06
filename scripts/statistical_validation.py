"""
scripts/statistical_validation.py
────────────────────────────────────────────────────────
Statistical validation for CHAAD project results.

Implements:
  1. Bootstrap confidence intervals (1000 resamples) for all metrics
  2. Multi-seed evaluation support
  3. McNemar's test for paired method comparison
  4. Wilcoxon signed-rank test for score distributions
  5. Effect size (Cohen's d) between methods

Usage:
    python scripts/statistical_validation.py \
        --predictions reports/test_predictions.csv \
        --baseline-predictions reports/baseline_predictions.csv \
        --output reports/statistical_validation_report.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve,
)
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seed import set_seed
from utils.split_utils import get_repo_commit


# ─────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class BootstrapCI:
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 1000

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self):
        return (f"{self.metric}: {self.point_estimate:.4f} "
                f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
                f"({self.ci_level*100:.0f}% CI)")


@dataclass
class StatisticalTest:
    test_name: str
    statistic: float
    p_value: float
    significant: bool        # p < 0.05
    effect_size: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiSeedResult:
    seeds: List[int]
    metric_name: str
    values: List[float]
    mean: float
    std: float
    sem: float
    ci_95: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "metric_name": self.metric_name,
            "values": self.values,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "ci_95_lower": self.ci_95[0],
            "ci_95_upper": self.ci_95[1],
        }


# ─────────────────────────────────────────────────────────
#  Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────

def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> List[BootstrapCI]:
    """
    Compute bootstrap confidence intervals for all key metrics.

    Uses stratified bootstrap to preserve class balance in each resample.
    """
    np.random.seed(random_state)
    n = len(y_true)

    # Separate indices by class for stratified bootstrap
    norm_idx = np.where(y_true == 0)[0]
    anom_idx = np.where(y_true == 1)[0]

    metrics_funcs = {
        "roc_auc": lambda t, s: roc_auc_score(t, s),
        "pr_auc": lambda t, s: average_precision_score(t, s),
        "p_auc_01": lambda t, s: roc_auc_score(t, s, max_fpr=0.1),
    }

    # Bootstrap samples
    bootstrap_results = {name: [] for name in metrics_funcs}
    bootstrap_thresholds = []

    for _ in range(n_bootstrap):
        # Stratified resample
        boot_norm = resample(norm_idx, replace=True, n_samples=len(norm_idx))
        boot_anom = resample(anom_idx, replace=True, n_samples=len(anom_idx))
        boot_idx = np.concatenate([boot_norm, boot_anom])

        t_boot = y_true[boot_idx]
        s_boot = y_scores[boot_idx]

        for name, func in metrics_funcs.items():
            try:
                val = func(t_boot, s_boot)
                bootstrap_results[name].append(val)
            except ValueError:
                bootstrap_results[name].append(np.nan)

        # Threshold via Youden's J
        try:
            fpr, tpr, thresh = roc_curve(t_boot, s_boot)
            j = tpr - fpr
            bootstrap_thresholds.append(float(thresh[np.argmax(j)]))
        except (ValueError, IndexError):
            bootstrap_thresholds.append(0.5)

    # Also compute point estimates for threshold-dependent metrics
    fpr_all, tpr_all, thresh_all = roc_curve(y_true, y_scores)
    j_all = tpr_all - fpr_all
    best_thresh = float(thresh_all[np.argmax(j_all)])
    y_pred = (y_scores >= best_thresh).astype(int)

    threshold_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Bootstrap threshold-dependent metrics
    boot_threshold_results = {name: [] for name in threshold_metrics}
    for b_idx in range(n_bootstrap):
        boot_norm = resample(norm_idx, replace=True, n_samples=len(norm_idx))
        boot_anom = resample(anom_idx, replace=True, n_samples=len(anom_idx))
        boot_idx = np.concatenate([boot_norm, boot_anom])

        t_boot = y_true[boot_idx]
        s_boot = y_scores[boot_idx]
        th_boot = bootstrap_thresholds[b_idx % len(bootstrap_thresholds)]
        p_boot = (s_boot >= th_boot).astype(int)

        try:
            boot_threshold_results["accuracy"].append(accuracy_score(t_boot, p_boot))
            boot_threshold_results["precision"].append(precision_score(t_boot, p_boot, zero_division=0))
            boot_threshold_results["recall"].append(recall_score(t_boot, p_boot, zero_division=0))
            boot_threshold_results["f1"].append(f1_score(t_boot, p_boot, zero_division=0))
        except ValueError:
            for k in threshold_metrics:
                boot_threshold_results[k].append(np.nan)

    # Compile results
    cis: List[BootstrapCI] = []

    alpha = 1.0 - ci_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    for name in metrics_funcs:
        vals = np.array(bootstrap_results[name])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            cis.append(BootstrapCI(
                metric=name,
                point_estimate=np.median(vals),
                ci_lower=float(np.percentile(vals, lower_pct)),
                ci_upper=float(np.percentile(vals, upper_pct)),
                ci_level=ci_level,
                n_bootstrap=n_bootstrap,
            ))

    for name, point_val in threshold_metrics.items():
        vals = np.array(boot_threshold_results[name])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            cis.append(BootstrapCI(
                metric=name,
                point_estimate=point_val,
                ci_lower=float(np.percentile(vals, lower_pct)),
                ci_upper=float(np.percentile(vals, upper_pct)),
                ci_level=ci_level,
                n_bootstrap=n_bootstrap,
            ))

    return cis


# ─────────────────────────────────────────────────────────
#  Statistical Tests Between Methods
# ─────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_method_a: np.ndarray,
    y_pred_method_b: np.ndarray,
) -> StatisticalTest:
    """
    McNemar's test for paired nominal data.
    Tests whether two methods have different error rates.

    H0: Both methods have the same error rate.
    """
    # Contingency table: count where methods agree/disagree
    both_correct = np.sum((y_pred_method_a == y_true) & (y_pred_method_b == y_true))
    a_correct_b_wrong = np.sum((y_pred_method_a == y_true) & (y_pred_method_b != y_true))
    a_wrong_b_correct = np.sum((y_pred_method_a != y_true) & (y_pred_method_b == y_true))
    both_wrong = np.sum((y_pred_method_a != y_true) & (y_pred_method_b != y_true))

    # McNemar uses discordant pairs only
    b = a_correct_b_wrong
    c = a_wrong_b_correct

    if b + c == 0:
        return StatisticalTest(
            test_name="McNemar",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="No discordant pairs found."
        )

    # McNemar statistic with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)

    return StatisticalTest(
        test_name="McNemar",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < 0.05,
        interpretation=(
            f"Statistically significant difference in error rates (p={p_value:.4f})"
            if p_value < 0.05 else
            f"No statistically significant difference (p={p_value:.4f})"
        ),
    )


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> StatisticalTest:
    """
    Wilcoxon signed-rank test for paired continuous scores.
    Tests whether the score distributions differ significantly.

    H0: The median difference between paired scores is zero.
    """
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)

    # Effect size: r = Z / sqrt(N)
    z_stat = stats.norm.ppf(p_value / 2) if p_value < 1.0 else 0.0
    effect_size = abs(z_stat) / np.sqrt(len(scores_a)) if len(scores_a) > 0 else 0.0

    return StatisticalTest(
        test_name="Wilcoxon Signed-Rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < 0.05,
        effect_size=float(effect_size),
        interpretation=(
            f"Score distributions differ significantly (p={p_value:.4f}, r={effect_size:.3f})"
            if p_value < 0.05 else
            f"Score distributions not significantly different (p={p_value:.4f})"
        ),
    )


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cohen's d effect size for independent samples.
    d = (mean_a - mean_b) / pooled_std
    """
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    n_a, n_b = len(scores_a), len(scores_b)

    # Pooled standard deviation
    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> StatisticalTest:
    """
    DeLong's test for comparing two correlated ROC-AUC values.

    Simplified implementation using bootstrap.
    H0: AUC_a = AUC_b
    """
    n = len(y_true)
    auc_diff_bootstrap = []

    np.random.seed(42)
    for _ in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            auc_a = roc_auc_score(y_true[idx], scores_a[idx])
            auc_b = roc_auc_score(y_true[idx], scores_b[idx])
            auc_diff_bootstrap.append(auc_a - auc_b)
        except ValueError:
            pass

    if len(auc_diff_bootstrap) < 100:
        return StatisticalTest(
            test_name="DeLong (bootstrap)",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Insufficient bootstrap samples.",
        )

    # Compute p-value from bootstrap distribution
    observed_diff = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    auc_diff_bootstrap = np.array(auc_diff_bootstrap)

    # Two-sided p-value
    p_value = np.mean(np.abs(auc_diff_bootstrap) >= np.abs(observed_diff))

    return StatisticalTest(
        test_name="DeLong (bootstrap)",
        statistic=float(observed_diff),
        p_value=float(p_value),
        significant=p_value < 0.05,
        effect_size=float(observed_diff) / (np.std(auc_diff_bootstrap) + 1e-8),
        interpretation=(
            f"ROC-AUC differs significantly (ΔAUC={observed_diff:.4f}, p={p_value:.4f})"
            if p_value < 0.05 else
            f"ROC-AUC not significantly different (ΔAUC={observed_diff:.4f}, p={p_value:.4f})"
        ),
    )


# ─────────────────────────────────────────────────────────
#  Multi-Seed Analysis
# ─────────────────────────────────────────────────────────

def multi_seed_analysis(
    seeds: List[int],
    results_per_seed: Dict[int, Dict[str, float]],
) -> List[MultiSeedResult]:
    """
    Analyze metric stability across multiple random seeds.

    Args:
        seeds: List of random seeds used
        results_per_seed: {seed: {metric_name: value}}
    """
    metrics = list(next(iter(results_per_seed.values())).keys())
    output = []

    for metric in metrics:
        values = [results_per_seed[s][metric] for s in seeds if metric in results_per_seed[s]]

        if len(values) < 2:
            continue

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(len(values))

        # 95% CI via t-distribution
        t_crit = stats.t.ppf(0.975, df=len(values) - 1)
        ci_lower = mean - t_crit * sem
        ci_upper = mean + t_crit * sem

        output.append(MultiSeedResult(
            seeds=seeds,
            metric_name=metric,
            values=values,
            mean=float(mean),
            std=float(std),
            sem=float(sem),
            ci_95=(float(ci_lower), float(ci_upper)),
        ))

    return output


# ─────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical validation for CHAAD")
    parser.add_argument("--predictions", help="CSV with predictions (must have true_label, anomaly_score)")
    parser.add_argument("--baseline-predictions", help="CSV with baseline predictions")
    parser.add_argument("--output", default="reports/statistical_validation_report.json")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    set_seed(42, deterministic_cudnn=False)

    print("=" * 70)
    print("  STATISTICAL VALIDATION REPORT")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_repo_commit(),
    }

    # ── Load prediction data ──────────────────────────────────
    if args.predictions and os.path.exists(args.predictions):
        df = pd.read_csv(args.predictions)
        y_true = (df["true_label"].values >= 0.5).astype(int)
        y_scores = df["anomaly_score"].values

        print(f"\nLoaded {len(y_true)} predictions")
        print(f"  Normal: {(y_true == 0).sum()}, Abnormal: {(y_true == 1).sum()}")

        # Bootstrap CIs
        print(f"\n── Bootstrap Confidence Intervals ({args.n_bootstrap} resamples, {args.ci_level*100:.0f}%) ──")
        cis = bootstrap_confidence_intervals(
            y_true, y_scores,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
        )
        for ci in cis:
            print(f"  {ci}")

        report["bootstrap_confidence_intervals"] = [ci.to_dict() for ci in cis]

        # Compare with baseline if provided
        if args.baseline_predictions and os.path.exists(args.baseline_predictions):
            df_baseline = pd.read_csv(args.baseline_predictions)
            if "anomaly_score" in df_baseline.columns:
                y_scores_baseline = df_baseline["anomaly_score"].values

                print("\n── Method Comparison Tests ──")

                # DeLong's test for AUC
                delong = delong_test(y_true, y_scores, y_scores_baseline)
                print(f"  {delong.test_name}: p={delong.p_value:.4f}, sig={delong.significant}")
                report["delong_test"] = delong.to_dict()

                # Wilcoxon test on scores
                wilcox = wilcoxon_test(y_scores, y_scores_baseline)
                print(f"  {wilcox.test_name}: p={wilcox.p_value:.4f}, sig={wilcox.significant}")
                report["wilcoxon_test"] = wilcox.to_dict()

                # Cohen's d
                d = cohens_d(y_scores, y_scores_baseline)
                print(f"  Cohen's d: {d:.4f}")
                report["cohens_d"] = float(d)

                # McNemar on thresholded predictions
                fpr, tpr, thresh = roc_curve(y_true, y_scores)
                best_t = float(thresh[np.argmax(tpr - fpr)])
                y_pred_a = (y_scores >= best_t).astype(int)
                y_pred_b = (y_scores_baseline >= best_t).astype(int)
                mcn = mcnemar_test(y_true, y_pred_a, y_pred_b)
                print(f"  {mcn.test_name}: p={mcn.p_value:.4f}, sig={mcn.significant}")
                report["mcnemar_test"] = mcn.to_dict()
    else:
        print("\n⚠ No prediction file provided. Skipping bootstrap and comparison tests.")
        print("  Usage: --predictions reports/test_predictions.csv")

    # ── Save ──────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
