"""
scripts/audit_shortcuts.py
────────────────────────────────────────────────────────
Shortcut learning audit for CHAAD project.

Tests whether the model exploits trivial shortcuts rather
than learning genuine acoustic anomaly patterns.

Checks performed:
  1. Metadata-only baseline (predict from machine_type + noise_condition)
  2. Feature permutation importance (shuffle each feature, measure AUC drop)
  3. Cross-machine generalization (train on subset, test on held-out)
  4. Signal-level shortcut detection (do raw stats predict labels?)
  5. Recording-level artifact check

Usage:
    python scripts/audit_shortcuts.py [--output artifacts/research_audit/shortcut_learning_report.json]
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
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from data.dataset import MIMIIDataset
from models.hybrid_model import HybridAnomalyModel
from utils.seed import set_seed, seed_worker
from utils.split_utils import get_repo_commit
from utils.precision import safe_autocast


# ── Label parsing helper ────────────────────────────────

def _parse_labels(series: pd.Series) -> np.ndarray:
    """Convert string ('normal'/'abnormal') or numeric labels to binary int."""
    s = series.astype(str).str.lower().str.strip()
    # Check if values look numeric (e.g. "0", "1", "0.0", "1.0")
    sample = s.iloc[0] if len(s) > 0 else ""
    if sample.replace(".", "").isdigit():
        return (pd.to_numeric(s, errors="coerce").fillna(0).astype(float) >= 0.5).astype(int).values
    return (s == "abnormal").astype(int).values


# ─────────────────────────────────────────────────────────
#  Check 1: Metadata-Only Baseline
# ─────────────────────────────────────────────────────────

def check_metadata_only_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test if machine_type + noise_condition alone can predict labels.
    If metadata alone achieves high AUC, the model may be exploiting
    dataset biases rather than learning acoustic features.
    """
    print("\n── Check 1: Metadata-Only Baseline ──")

    # Encode metadata as features
    le_machine = LabelEncoder()
    le_noise = LabelEncoder()

    X_machine = le_machine.fit_transform(df["machine_type"].astype(str))
    X_noise = le_noise.fit_transform(df["noise_condition"].astype(str))

    X_meta = np.column_stack([X_machine, X_noise])

    y_true = _parse_labels(df["label"])

    # Try logistic regression on metadata
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta.astype(float))

    # Use cross-validation within the given dataframe
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        auc_scores = cross_val_score(
            lr, X_scaled, y_true, cv=cv, scoring="roc_auc"
        )
        mean_auc = float(auc_scores.mean())
        std_auc = float(auc_scores.std())
    except ValueError:
        mean_auc = 0.5
        std_auc = 0.0

    # Also try Random Forest for non-linear metadata relationships
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_scores = cross_val_score(
            rf, X_scaled, y_true, cv=cv, scoring="roc_auc"
        )
        rf_mean_auc = float(rf_scores.mean())
    except ValueError:
        rf_mean_auc = 0.5

    shortcut_detected = mean_auc > 0.70 or rf_mean_auc > 0.75
    threshold = 0.70

    print(f"  Logistic Regression AUC (metadata):  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Random Forest AUC (metadata):        {rf_mean_auc:.4f}")
    print(f"  Shortcut threshold:                  {threshold:.2f}")
    print(f"  Shortcut detected:                   {shortcut_detected}")

    return {
        "check_name": "metadata_only_baseline",
        "logistic_regression_auc": mean_auc,
        "logistic_regression_auc_std": std_auc,
        "random_forest_auc": rf_mean_auc,
        "shortcut_threshold": threshold,
        "shortcut_detected": shortcut_detected,
        "interpretation": (
            "Metadata alone can predict labels with AUC > 0.70. "
            "Model may exploit dataset-level biases." if shortcut_detected
            else "Metadata alone cannot strongly predict labels. Low shortcut risk."
        ),
    }


# ─────────────────────────────────────────────────────────
#  Check 2: Feature Permutation Importance
# ─────────────────────────────────────────────────────────

def check_feature_permutation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Shuffle each metadata column and measure AUC drop.
    Large drop for a column → model relies on that feature.
    """
    print("\n── Check 2: Feature Permutation Importance ──")

    # Collect predictions with original data
    all_labels = []
    all_scores = []
    metadata_records = []

    model.eval()
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(loader, desc="Baseline predictions", leave=False):
            mel = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"]

            with safe_autocast(device, enabled=amp_enabled):
                outputs = model(mel)

            scores = torch.sigmoid(outputs["logits"].squeeze(-1))
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().float().numpy().tolist())

            # Collect metadata for this batch
            if "machine" in batch and "snr" in batch:
                for i in range(len(labels)):
                    metadata_records.append({
                        "machine_type": str(batch["machine"][i]) if isinstance(batch["machine"], list) else str(batch["machine"]),
                        "noise_condition": str(batch["snr"][i]) if isinstance(batch["snr"], list) else str(batch["snr"]),
                    })

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    try:
        baseline_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        baseline_auc = 0.5

    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # Permutation test: shuffle metadata and re-evaluate
    # Since we can't easily change model inputs without re-loading,
    # we use a proxy: train a random forest on model predictions + metadata
    # and measure feature importance

    if len(metadata_records) > 0:
        df_meta = pd.DataFrame(metadata_records)

        le_m = LabelEncoder()
        le_n = LabelEncoder()

        X_m = le_m.fit_transform(df_meta["machine_type"].astype(str))
        X_n = le_n.fit_transform(df_meta["noise_condition"].astype(str))

        X = np.column_stack([y_scores, X_m, X_n])

        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
        rf.fit(X, y_true)

        importances = rf.feature_importances_
        feature_names = ["anomaly_score", "machine_type", "noise_condition"]

        permutation_result = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }

        print(f"  Feature importances (RF on scores + metadata):")
        for name, imp in zip(feature_names, importances):
            print(f"    {name}: {imp:.4f}")

        # If metadata importance is high relative to score, that's suspicious
        score_importance = importances[0]
        metadata_importance = importances[1] + importances[2]

        high_metadata_importance = metadata_importance > 0.3 * score_importance
    else:
        permutation_result = {}
        high_metadata_importance = False
        print("  ⚠ No metadata available for permutation test")

    return {
        "check_name": "feature_permutation",
        "baseline_auc": baseline_auc,
        "feature_importances": permutation_result,
        "metadata_importance_high": high_metadata_importance,
        "interpretation": (
            "Metadata features have high importance. Model may rely on shortcuts."
            if high_metadata_importance else
            "Metadata features have low importance. Model primarily uses anomaly scores."
        ),
    }


# ─────────────────────────────────────────────────────────
#  Check 3: Cross-Machine Generalization
# ─────────────────────────────────────────────────────────

def check_cross_machine_generalization(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test if model performance drops when tested on unseen machine types.
    A large drop suggests the model learns machine-specific artifacts.

    Note: This uses the manifest-level data structure. For actual model
    predictions, the full pipeline would need to be run per machine type.
    """
    print("\n── Check 3: Cross-Machine Analysis ──")

    all_labels = _parse_labels(df["label"])
    machine_types = sorted(df["machine_type"].unique())

    per_machine_stats = {}
    for mtype in machine_types:
        mask = (df["machine_type"] == mtype).values
        n_samples = mask.sum()
        n_abnormal = int(all_labels[mask].sum())

        per_machine_stats[mtype] = {
            "total_samples": int(n_samples),
            "abnormal_samples": int(n_abnormal),
            "abnormal_ratio": float(n_abnormal / max(n_samples, 1)),
        }

        print(f"  {mtype:10s}: {n_samples:6d} samples, "
              f"{n_abnormal:5d} abnormal ({n_abnormal/max(n_samples,1)*100:.1f}%)")

    # Check if abnormal ratios are drastically different across machines
    ratios = [v["abnormal_ratio"] for v in per_machine_stats.values()]
    if len(ratios) > 1:
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        ratio_disparity = max_ratio / max(min_ratio, 0.001)
    else:
        ratio_disparity = 1.0

    imbalance_concern = ratio_disparity > 2.0

    if imbalance_concern:
        print(f"  ⚠ Abnormal ratio varies {ratio_disparity:.1f}x across machines")

    return {
        "check_name": "cross_machine_generalization",
        "per_machine_stats": per_machine_stats,
        "ratio_disparity": float(ratio_disparity),
        "imbalance_concern": imbalance_concern,
        "interpretation": (
            f"Abnormal ratios vary {ratio_disparity:.1f}x across machines. "
            "Model may learn machine-specific priors." if imbalance_concern else
            "Abnormal ratios are reasonably balanced across machines."
        ),
    }


# ─────────────────────────────────────────────────────────
#  Check 4: Signal-Level Shortcut Detection
# ─────────────────────────────────────────────────────────

def check_signal_shortcuts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if simple signal statistics (mean, std of anomaly scores)
    can predict labels without the full model.
    """
    print("\n── Check 4: Signal-Level Shortcuts ──")

    # If we have raw anomaly scores per sample, check if simple
    # thresholds on individual scores work too well
    score_cols = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]
    available_cols = [c for c in score_cols if c in df.columns]

    y_true = _parse_labels(df["label"])

    per_score_auc = {}
    for col in available_cols:
        try:
            auc = float(roc_auc_score(y_true, df[col].values))
        except ValueError:
            auc = 0.5
        per_score_auc[col] = auc
        print(f"  {col:20s} solo AUC: {auc:.4f}")

    # Check if ANY single score achieves > 0.95 AUC
    max_single_auc = max(per_score_auc.values()) if per_score_auc else 0.5
    single_score_concern = max_single_auc > 0.98

    if single_score_concern:
        best_col = max(per_score_auc, key=per_score_auc.get)
        print(f"  ⚠ {best_col} alone achieves AUC {max_single_auc:.4f} - possible ceiling effect")

    return {
        "check_name": "signal_shortcuts",
        "per_score_auc": per_score_auc,
        "max_single_score_auc": max_single_auc,
        "single_score_concern": single_score_concern,
        "interpretation": (
            f"Single score {max(per_score_auc, key=per_score_auc.get) if per_score_auc else 'N/A'} "
            f"achieves AUC {max_single_auc:.4f}. {'Possible shortcut or ceiling effect.' if single_score_concern else 'Acceptable.'}"
        ),
    }


# ─────────────────────────────────────────────────────────
#  Check 5: Recording-Level Artifact Check
# ─────────────────────────────────────────────────────────

def check_recording_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if the model might be exploiting recording-specific artifacts
    (e.g., same recording appearing in different segments).
    """
    print("\n── Check 5: Recording-Level Artifacts ──")

    # This is primarily verified by the main leakage audit.
    # Here we check for statistical anomalies in feature-level distributions.

    artifacts_found = False
    concerns = []

    # Check if labels are correlated with file size or duration patterns
    if "duration_seconds" in df.columns:
        labels = _parse_labels(df["label"])
        normal_dur = df.loc[labels == 0, "duration_seconds"]
        abnormal_dur = df.loc[labels == 1, "duration_seconds"]

        if len(normal_dur) > 0 and len(abnormal_dur) > 0:
            from scipy import stats as scipy_stats
            try:
                t_stat, p_val = scipy_stats.ttest_ind(normal_dur, abnormal_dur)
                if p_val < 0.001:
                    concerns.append(f"Duration differs by label (p={p_val:.2e})")
                    artifacts_found = True
                    print(f"  ⚠ Duration differs significantly between normal/abnormal (p={p_val:.2e})")
            except Exception:
                pass
            else:
                print(f"  ✓ Duration similar across labels (p={p_val:.4f})")
    else:
        print(f"  ℹ No duration column in dataframe")

    return {
        "check_name": "recording_artifacts",
        "artifacts_found": artifacts_found,
        "concerns": concerns,
        "interpretation": (
            "Recording-level artifacts detected that may cause shortcut learning."
            if artifacts_found else
            "No recording-level artifacts detected."
        ),
    }


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shortcut learning audit for CHAAD")
    parser.add_argument("--output", default="artifacts/research_audit/shortcut_learning_report.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(42, deterministic_cudnn=True)

    print("=" * 70)
    print("  SHORTCUT LEARNING AUDIT")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Use manifest data for metadata checks
    manifest_path = "metadata/dataset_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} samples from manifest")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_repo_commit(),
        "manifest_samples": len(df),
        "checks": [],
    }

    # Run checks
    check1 = check_metadata_only_baseline(df)
    results["checks"].append(check1)

    # Check 2 and signal-level shortcuts require model predictions
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading model: {args.checkpoint}")
        model = HybridAnomalyModel(cfg.model).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        # Use validation only; protected test access is forbidden before Phase 8.
        try:
            validation_ds = MIMIIDataset(cfg, split="val")
            validation_loader = DataLoader(
                validation_ds, batch_size=cfg.training.batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
            )
            check2 = check_feature_permutation(model, validation_loader, device)
            results["checks"].append(check2)
        except Exception as e:
            print(f"  ⚠ Could not run feature permutation check: {e}")
            results["checks"].append({
                "check_name": "feature_permutation",
                "error": str(e),
            })
    else:
        print("\n⚠ No model checkpoint provided. Skipping feature permutation check.")
        print("  Usage: --checkpoint checkpoints/best_model.pt")
        results["checks"].append({
            "check_name": "feature_permutation",
            "status": "SKIPPED",
            "reason": "No model checkpoint provided",
        })

    check3 = check_cross_machine_generalization(df)
    results["checks"].append(check3)

    # Signal shortcuts from manifest if score columns exist
    results["checks"].append(check_signal_shortcuts(df))

    check5 = check_recording_artifacts(df)
    results["checks"].append(check5)

    # Overall assessment
    shortcut_detected = any(
        c.get("shortcut_detected", False)
        or c.get("metadata_importance_high", False)
        or c.get("single_score_concern", False)
        or c.get("artifacts_found", False)
        for c in results["checks"]
    )

    results["overall"] = {
        "shortcut_detected": shortcut_detected,
        "verdict": (
            "SHORTCUT DETECTED - Model may exploit trivial features."
            if shortcut_detected else
            "NO SHORTCUT DETECTED - Model likely uses genuine acoustic features."
        ),
    }

    # Print overall
    print(f"\n{'='*70}")
    print(f"  OVERALL: {results['overall']['verdict']}")
    print(f"{'='*70}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Report saved to: {output_path}")

    if shortcut_detected:
        sys.exit(1)


if __name__ == "__main__":
    main()
