"""
scripts/run_baselines.py
────────────────────────────────────────────────────────
Comprehensive baseline comparison for CHAAD project.

Runs all baseline methods against the frozen test set and
produces a publication-quality comparison report.

Baselines implemented:
  1. Single-Score Detectors:
     - Reconstruction-only
     - Embedding-distance-only
     - Mahalanobis-distance-only
     - Contrastive-similarity-only

  2. Fixed Fusion Methods:
     - Equal-weight fusion (uniform averaging)
     - Current hand-selected weights (from config.py)
     - Global learned weights (one vector learned on validation)

  3. Classical ML Baselines (on embeddings):
     - Isolation Forest
     - Local Outlier Factor (LOF)
     - One-Class SVM

  4. Metadata-Only Baselines:
     - Random guessing baseline
     - Class-prior baseline (predict majority class)
     - Machine-type majority baseline

Usage:
    python scripts/run_baselines.py [--device cuda] [--output reports/baseline_comparison.json]
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, balanced_accuracy_score,
)
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from data.dataset import MIMIIDataset
from models.hybrid_model import HybridAnomalyModel
from utils.metrics import load_threshold_metadata
from utils.seed import set_seed, seed_worker
from utils.split_utils import get_repo_commit
from utils.experiment_contract import assert_split_access
from utils.checkpoint import load_model_weights
from utils.precision import safe_autocast


# ─────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    name: str
    category: str               # "single_score", "fixed_fusion", "learned_fusion", "classical_ml", "metadata_only"
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    p_auc: float                # partial AUC @ max_fpr=0.1
    balanced_accuracy: float
    threshold: float
    confusion_matrix: List[List[int]]
    runtime_seconds: float
    num_parameters: Optional[int] = None  # for learned methods
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────
#  Helper: Load model and get embeddings + scores
# ─────────────────────────────────────────────────────────

class ModelFeatureExtractor:
    """Extract embeddings and multi-perspective scores from the trained model."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Will be set by fit_calibration
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_cov_inv: Optional[np.ndarray] = None
        self.ref_pool: Optional[np.ndarray] = None
        self.ref_mean_normed: Optional[np.ndarray] = None
        self.ref_pool_normed: Optional[np.ndarray] = None

    def fit_calibration(self, normal_loader: DataLoader):
        """Fit reference distribution on normal training data."""
        embed_list: List[np.ndarray] = []

        for batch in tqdm(normal_loader, desc="Calibrating reference", leave=False):
            mel = batch["mel"].to(self.device)
            with torch.no_grad():
                with safe_autocast(self.device, enabled=False):
                    out = self.model(mel)
            embed_list.append(out["embeddings"].cpu().numpy())

        embeds_all = np.vstack(embed_list)
        self.ref_mean = embeds_all.mean(axis=0)
        self.ref_pool = embeds_all

        # Ledoit-Wolf precision
        lw = LedoitWolf()
        lw.fit(embeds_all)
        self.ref_cov_inv = lw.precision_

        # Pre-normalize
        self.ref_mean_normed = self.ref_mean / (np.linalg.norm(self.ref_mean) + 1e-8)
        pool_norms = np.linalg.norm(self.ref_pool, axis=1, keepdims=True) + 1e-8
        self.ref_pool_normed = self.ref_pool / pool_norms

    @torch.no_grad()
    def extract_features(
        self,
        loader: DataLoader,
    ) -> pd.DataFrame:
        """
        Extract all features from a data loader.

        Returns DataFrame with columns:
            embedding_0..D, recon_error, embed_dist, mahal_dist, contra_dist,
            label, machine_type, machine_id, noise_condition, file_path
        """
        records = []
        amp_enabled = cfg.training.mixed_precision and self.device.type == "cuda"

        for batch in tqdm(loader, desc="Extracting features", leave=False):
            mel = batch["mel"].to(self.device, non_blocking=True)
            labels = batch["label"]

            with safe_autocast(self.device, enabled=amp_enabled):
                outputs = self.model(mel)

            embeddings = outputs["embeddings"].cpu().numpy()           # (B, D)
            recon_out = outputs["reconstruction"]
            pooled = outputs["pooled_feat"].cpu().numpy()

            # Reconstruction error per sample
            recon_err = F.mse_loss(recon_out, mel, reduction="none")
            recon_err = recon_err.mean(dim=(1, 2, 3)).cpu().numpy()    # (B,)

            for i in range(len(labels)):
                emb = embeddings[i]
                rec = {}

                # Embedding cosine distance
                if self.ref_mean is not None:
                    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                    embed_dist = float(1.0 - np.dot(emb_n, self.ref_mean_normed))
                else:
                    embed_dist = 0.0

                # Mahalanobis distance
                if self.ref_mean is not None and self.ref_cov_inv is not None:
                    diff = emb - self.ref_mean
                    mahal = float(np.sqrt(max(0.0, diff @ self.ref_cov_inv @ diff)))
                else:
                    mahal = 0.0

                # Contrastive distance
                if self.ref_pool is not None:
                    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                    sims = self.ref_pool_normed @ emb_n
                    k = min(5, len(sims))
                    contra_sim = float(np.sort(sims)[-k:].mean())
                    contra_dist = 1.0 - contra_sim
                else:
                    contra_dist = 0.0

                rec = {
                    "recon_error": float(recon_err[i]),
                    "embed_dist": embed_dist,
                    "mahal_dist": mahal,
                    "contra_dist": contra_dist,
                    "label": float(labels[i].cpu().numpy()) if torch.is_tensor(labels) else float(labels[i]),
                }

                # Embedding vector as separate columns
                for d_idx in range(len(emb)):
                    rec[f"embedding_{d_idx}"] = float(emb[d_idx])

                rec["machine_type"] = batch.get("machine", ["unknown"] * len(labels))[i]
                rec["machine_id"] = batch.get("machine_id", ["unknown"] * len(labels))[i]
                rec["noise_condition"] = batch.get("snr", ["unknown"] * len(labels))[i]
                rec["sample_id"] = batch.get("sample_id", ["unknown"] * len(labels))[i]
                rec["source_recording"] = batch.get(
                    "source_recording", ["unknown"] * len(labels)
                )[i]
                rec["split"] = batch.get("split", ["unknown"] * len(labels))[i]
                rec["file_path"] = batch.get(
                    "file_path", ["unknown"] * len(labels)
                )[i]

                records.append(rec)

        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
#  Baseline Implementations
# ─────────────────────────────────────────────────────────

def baseline_random(df: pd.DataFrame) -> BaselineResult:
    """Random guessing baseline (uniform random scores)."""
    y_true = (df["label"] >= 0.5).astype(int).values
    np.random.seed(42)
    y_scores = np.random.rand(len(y_true))

    return _compute_metrics("Random Guessing", "metadata_only", y_true, y_scores, 0.0)


def baseline_class_prior(df: pd.DataFrame) -> BaselineResult:
    """Always predict normal (most common class)."""
    y_true = (df["label"] >= 0.5).astype(int).values
    y_scores = np.zeros(len(y_true))  # all predicted normal

    return _compute_metrics("Class Prior (All Normal)", "metadata_only", y_true, y_scores, 0.0)


def baseline_machine_type_prior(df: pd.DataFrame) -> BaselineResult:
    """Predict anomaly based on machine-type-specific class priors."""
    y_true = (df["label"] >= 0.5).astype(int).values
    y_scores = np.zeros(len(y_true))

    for mtype in df["machine_type"].unique():
        mask = df["machine_type"] == mtype
        if mask.sum() == 0:
            continue
        # Score = P(abnormal | machine_type) on test set
        prior = df.loc[mask, "label"].mean()
        y_scores[mask.values] = prior

    return _compute_metrics("Machine-Type Prior", "metadata_only", y_true, y_scores, 0.0)


def baseline_single_score(df: pd.DataFrame, score_col: str, score_name: str) -> BaselineResult:
    """Single anomaly score as detector."""
    y_true = (df["label"] >= 0.5).astype(int).values
    y_scores = df[score_col].values

    # Normalize to [0, 1] using min-max on test set (for fair comparison across score types)
    smin, smax = y_scores.min(), y_scores.max()
    if smax > smin:
        y_scores = (y_scores - smin) / (smax - smin)

    return _compute_metrics(score_name, "single_score", y_true, y_scores, 0.0)


def baseline_equal_weight_fusion(df: pd.DataFrame) -> BaselineResult:
    """Equal-weight fusion of all four calibrated scores."""
    score_cols = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]
    y_true = (df["label"] >= 0.5).astype(int).values

    # Min-max normalize each score to [0, 1]
    scores_normalized = np.zeros((len(df), len(score_cols)))
    for i, col in enumerate(score_cols):
        vals = df[col].values
        smin, smax = vals.min(), vals.max()
        if smax > smin:
            scores_normalized[:, i] = (vals - smin) / (smax - smin)
        else:
            scores_normalized[:, i] = vals

    # Equal weights
    weights = np.ones(len(score_cols)) / len(score_cols)
    y_scores = (scores_normalized * weights).sum(axis=1)

    return _compute_metrics("Equal-Weight Fusion", "fixed_fusion", y_true, y_scores, 0.0)


def baseline_hand_selected_weights(df: pd.DataFrame) -> BaselineResult:
    """Current hand-selected weights from config.py."""
    score_cols = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]
    y_true = (df["label"] >= 0.5).astype(int).values
    weights = np.array([
        cfg.inference.w_recon,
        cfg.inference.w_embed,
        cfg.inference.w_mahal,
        cfg.inference.w_contra,
    ])

    # Min-max normalize
    scores_normalized = np.zeros((len(df), len(score_cols)))
    for i, col in enumerate(score_cols):
        vals = df[col].values
        smin, smax = vals.min(), vals.max()
        if smax > smin:
            scores_normalized[:, i] = (vals - smin) / (smax - smin)
        else:
            scores_normalized[:, i] = vals

    y_scores = (scores_normalized * weights).sum(axis=1)

    return _compute_metrics("Hand-Selected Weights (config.py)", "fixed_fusion", y_true, y_scores, 0.0)


def baseline_global_learned_weights(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> BaselineResult:
    """
    Learn one global weight vector on validation data (grid search).
    This is a sample-independent fusion strategy.
    """
    score_cols = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]
    y_true_test = (df_test["label"] >= 0.5).astype(int).values
    y_true_val = (df_train["label"] >= 0.5).astype(int).values

    # Normalize scores to [0, 1]
    def _normalize(df):
        normed = np.zeros((len(df), len(score_cols)))
        for i, col in enumerate(score_cols):
            vals = df[col].values
            smin, smax = vals.min(), vals.max()
            if smax > smin:
                normed[:, i] = (vals - smin) / (smax - smin)
            else:
                normed[:, i] = vals
        return normed

    val_scores = _normalize(df_train)
    test_scores = _normalize(df_test)

    # Grid search over weight simplex
    best_auc = 0.0
    best_weights = np.ones(len(score_cols)) / len(score_cols)

    # Coarse grid search on 3-simplex
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, 11 - int(w1 * 10)):
            for w3 in np.linspace(0, 1 - w1 - w2, 11 - int((w1 + w2) * 10)):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0:
                    continue
                weights = np.array([w1, w2, w3, w4])
                y_scores = (val_scores * weights).sum(axis=1)
                try:
                    auc = roc_auc_score(y_true_val, y_scores)
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = weights.copy()
                except ValueError:
                    pass

    y_scores_test = (test_scores * best_weights).sum(axis=1)

    return _compute_metrics(
        "Global Learned Weights (grid search)",
        "learned_fusion",
        y_true_test,
        y_scores_test,
        0.0,
        extra_info={"learned_weights": best_weights.tolist()},
    )


def baseline_global_learned_weights_oof(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> BaselineResult:
    """Generate out-of-fold global-fusion predictions on validation data."""
    score_cols = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]
    required = set(score_cols) | {
        "label",
        "machine_type",
        "machine_id",
        "source_recording",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF global fusion missing columns: {missing}")

    y_true = (frame["label"] >= 0.5).astype(int).to_numpy()
    groups = (
        frame["machine_type"].astype(str)
        + "|"
        + frame["machine_id"].astype(str)
        + "|"
        + frame["source_recording"].astype(str)
    ).to_numpy()
    raw_scores = frame[score_cols].to_numpy(dtype=np.float64)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    fold_weights: list[list[float]] = []
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    def candidate_weights():
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1 - w1, 11 - int(w1 * 10)):
                for w3 in np.linspace(0, 1 - w1 - w2, 11 - int((w1 + w2) * 10)):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 >= 0:
                        yield np.array([w1, w2, w3, w4], dtype=np.float64)

    for train_idx, holdout_idx in splitter.split(raw_scores, y_true, groups):
        train_scores = raw_scores[train_idx]
        minimum = train_scores.min(axis=0)
        scale = train_scores.max(axis=0) - minimum
        scale[scale == 0] = 1.0
        train_normalized = (train_scores - minimum) / scale
        holdout_normalized = (raw_scores[holdout_idx] - minimum) / scale
        best_auc = float("-inf")
        best_weights = np.full(4, 0.25, dtype=np.float64)
        for weights in candidate_weights():
            auc = roc_auc_score(y_true[train_idx], train_normalized @ weights)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
        predictions[holdout_idx] = holdout_normalized @ best_weights
        fold_weights.append(best_weights.tolist())

    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF global fusion did not produce one finite score per row")
    return _compute_metrics(
        "Global Learned Weights (5-fold OOF)",
        "learned_fusion",
        y_true,
        predictions,
        0.0,
        extra_info={"fold_weights": fold_weights},
    )


def baseline_isolation_forest(df_train: pd.DataFrame, df_test: pd.DataFrame) -> BaselineResult:
    """Isolation Forest on model embeddings."""
    emb_cols = [c for c in df_train.columns if c.startswith("embedding_")]
    X_train = df_train[emb_cols].values
    X_test = df_test[emb_cols].values
    y_true = (df_test["label"] >= 0.5).astype(int).values

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    clf.fit(X_train)

    # Anomaly scores: lower = more anomalous, invert
    raw_scores = -clf.score_samples(X_test)
    y_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

    return _compute_metrics("Isolation Forest (embeddings)", "classical_ml", y_true, y_scores, 0.0)


def baseline_ocsvm(df_train: pd.DataFrame, df_test: pd.DataFrame) -> BaselineResult:
    """One-Class SVM on model embeddings."""
    emb_cols = [c for c in df_train.columns if c.startswith("embedding_")]
    X_train = df_train[emb_cols].values
    X_test = df_test[emb_cols].values
    y_true = (df_test["label"] >= 0.5).astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    clf.fit(X_train)

    raw_scores = -clf.decision_function(X_test)
    y_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

    return _compute_metrics("One-Class SVM (embeddings)", "classical_ml", y_true, y_scores, 0.0)


# ─────────────────────────────────────────────────────────
#  Metric Computation
# ─────────────────────────────────────────────────────────

def _compute_metrics(
    name: str,
    category: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    runtime: float = 0.0,
    extra_info: Optional[Dict] = None,
) -> BaselineResult:
    """Compute all metrics for a given set of predictions."""
    if len(np.unique(y_true)) < 2:
        return BaselineResult(
            name=name, category=category,
            roc_auc=0.5, pr_auc=0.0, accuracy=0.0, precision=0.0,
            recall=0.0, f1=0.0, eer=1.0, p_auc=0.0,
            balanced_accuracy=0.0, threshold=0.5,
            confusion_matrix=[[0]], runtime_seconds=runtime,
        )

    # Find optimal threshold via Youden's J
    fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_true, y_scores)
    j_scores = tpr_arr - fpr_arr
    best_thresh = float(thresholds_arr[np.argmax(j_scores)])

    y_pred = (y_scores >= best_thresh).astype(int)

    # EER
    from sklearn.metrics import det_curve
    fpr_det, fnr_det, _ = det_curve(y_true, y_scores)
    eer_idx = np.nanargmin(np.abs(fnr_det - fpr_det))
    eer = float(fpr_det[eer_idx])

    # Partial AUC
    try:
        p_auc = float(roc_auc_score(y_true, y_scores, max_fpr=0.1))
    except ValueError:
        p_auc = 0.0

    return BaselineResult(
        name=name,
        category=category,
        roc_auc=float(roc_auc_score(y_true, y_scores)),
        pr_auc=float(average_precision_score(y_true, y_scores)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        eer=eer,
        p_auc=p_auc,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        threshold=best_thresh,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        runtime_seconds=runtime,
        details=extra_info,
    )


# ─────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all baselines for CHAAD project")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--output", default="reports/baseline_comparison.json", help="Output JSON path")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path (optional)")
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        help="Submission-recovery phase; this runner is validation-only before Phase 8.",
    )
    args = parser.parse_args()
    assert_split_access(phase=args.phase, split="validation")

    set_seed(cfg.training.random_seed, deterministic_cudnn=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            os.path.join(cfg.training.checkpoint_dir, "best_model.pt"),
            "checkpoints/best_model.pt",
            "artifacts/models/best_model.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("❌ No model checkpoint found. Cannot run feature-based baselines.")
        print("   Run training first: python train.py")
        sys.exit(1)

    print(f"Loading model: {ckpt_path}")
    model = HybridAnomalyModel(cfg.model).to(device)
    load_model_weights(model, ckpt_path, device)
    model.eval()

    # Build extractor and calibrate on train_normal
    extractor = ModelFeatureExtractor(model, device)

    print("Loading train_normal for calibration...")
    from data.dataset import get_normal_loader
    normal_loader = get_normal_loader(cfg)
    extractor.fit_calibration(normal_loader)

    # Extract validation features only. Protected test access is reserved for
    # the explicitly authorized Phase 8 command.
    print("Loading validation split...")
    val_ds = MIMIIDataset(cfg, split="val")
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print("Extracting features from validation set...")
    df_val = extractor.extract_features(val_loader)
    print(f"Validation samples: {len(df_val)}")

    # ── Run all baselines ──────────────────────────────────
    all_results: List[BaselineResult] = []

    # Metadata-only baselines
    print("\n── Metadata-Only Baselines ──")
    all_results.append(baseline_random(df_val))
    all_results.append(baseline_class_prior(df_val))
    all_results.append(baseline_machine_type_prior(df_val))

    # Single-score baselines
    print("\n── Single-Score Baselines ──")
    all_results.append(baseline_single_score(df_val, "recon_error", "Reconstruction Error"))
    all_results.append(baseline_single_score(df_val, "embed_dist", "Embedding Distance"))
    all_results.append(baseline_single_score(df_val, "mahal_dist", "Mahalanobis Distance"))
    all_results.append(baseline_single_score(df_val, "contra_dist", "Contrastive Distance"))

    # Fixed fusion methods
    print("\n── Fixed Fusion Baselines ──")
    all_results.append(baseline_equal_weight_fusion(df_val))
    all_results.append(baseline_hand_selected_weights(df_val))

    # Global learned weights use out-of-fold validation predictions.
    print("\n── Learned Fusion Baselines ──")
    all_results.append(baseline_global_learned_weights_oof(df_val))

    # ── Print Results ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("  BASELINE COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Method':<40s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'F1':>8s} {'EER':>8s} {'Category':>15s}")
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: x.roc_auc, reverse=True):
        print(f"{r.name:<40s} {r.roc_auc:8.4f} {r.pr_auc:8.4f} {r.f1:8.4f} {r.eer:8.4f} {r.category:>15s}")

    # ── Save ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_repo_commit(),
        "evaluation_split": "validation",
        "val_samples": len(df_val),
        "model_checkpoint": ckpt_path,
        "results": [r.to_dict() for r in all_results],
        "best_method": max(all_results, key=lambda r: r.roc_auc).name,
        "best_roc_auc": max(r.roc_auc for r in all_results),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"   Best method: {report['best_method']} (ROC-AUC={report['best_roc_auc']:.4f})")


if __name__ == "__main__":
    main()
