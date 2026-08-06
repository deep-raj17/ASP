"""
Independent validation-prediction export and evaluation audit for EXP-CHAAD-001.

This script regenerates validation predictions from the preserved checkpoint
without training, validates stable sample identifiers, compares two batch sizes
for deterministic export behavior, and recomputes validation-only metrics.

Usage:
    python scripts/audit_evaluation_pipeline.py --batch-sizes 16 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    det_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import cfg
from data.dataset import MIMIIDataset
from models.hybrid_model import HybridAnomalyModel


EXPECTED_CHECKPOINT_SHA256 = (
    "7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9"
)
EXPERIMENT_ID = "EXP-CHAAD-001"
DEFAULT_OUTPUT_DIR = Path("artifacts") / EXPERIMENT_ID
ORIGINAL_CORRUPTED_EXPORT = DEFAULT_OUTPUT_DIR / "validation_predictions.csv"


def configure_reproducible_inference() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def repo_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_and_model(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> tuple[HybridAnomalyModel | None, torch.device | None, str | None]:
    """Load the preserved checkpoint and verify its SHA-256 digest."""
    print("=" * 60)
    print("CHECKPOINT LOADING VERIFICATION")
    print("=" * 60)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"FAIL: checkpoint not found: {checkpoint_path}")
        return None, None, None

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != expected_sha256:
        print("FAIL: checkpoint hash mismatch")
        print(f"Expected: {expected_sha256}")
        print(f"Actual:   {checkpoint_hash}")
        return None, None, checkpoint_hash

    print(f"PASS: checkpoint SHA-256 verified: {checkpoint_hash}")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridAnomalyModel(cfg.model).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in state:
        print("FAIL: checkpoint has no model_state_dict key")
        return None, None, checkpoint_hash

    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("PASS: model state loaded and set to eval mode")
    return model, device, checkpoint_hash


def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().item()
    if isinstance(value, np.ndarray):
        return value[index].item()
    return value[index]


def generate_predictions_from_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    shuffle: bool = False,
) -> pd.DataFrame:
    """Generate predictions using sample IDs supplied by the dataset itself."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Batch size {batch_size}")):
            mel = batch["mel"].to(device)
            labels = batch["label"]
            outputs = model(mel)
            scores = torch.sigmoid(outputs["logits"].squeeze(-1))

            for row_idx in range(len(labels)):
                rows.append(
                    {
                        "sample_id": str(_batch_value(batch, "sample_id", row_idx)),
                        "true_label": float(_batch_value(batch, "label", row_idx)),
                        "predicted_score": float(scores[row_idx].detach().cpu().item()),
                        "machine_type": str(_batch_value(batch, "machine", row_idx)),
                        "machine_id": str(_batch_value(batch, "machine_id", row_idx)),
                        "snr": str(_batch_value(batch, "snr", row_idx)),
                        "source_path": str(_batch_value(batch, "file_path", row_idx)),
                        "relative_path": str(_batch_value(batch, "relative_path", row_idx)),
                        "split": str(_batch_value(batch, "split", row_idx)),
                        "batch_idx": batch_idx,
                        "sample_idx_in_batch": row_idx,
                    }
                )

    return pd.DataFrame(rows)


def generate_validation_predictions(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    print("\n" + "=" * 60)
    print(f"VALIDATION PREDICTION EXPORT: batch_size={batch_size}")
    print("=" * 60)
    val_ds = MIMIIDataset(cfg, split="val")
    df = generate_predictions_from_dataset(model, val_ds, device, batch_size=batch_size)
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Exported prediction rows: {len(df)}")
    return df, len(val_ds)


def _count_invalid_labels(labels: Iterable[Any]) -> int:
    invalid = 0
    for label in labels:
        try:
            if float(label) not in {0.0, 1.0}:
                invalid += 1
        except Exception:
            invalid += 1
    return invalid


def build_export_validation_report(
    df: pd.DataFrame,
    expected_count: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    status: str | None = None,
) -> dict[str, Any]:
    missing_id_count = int(df["sample_id"].isna().sum() + (df["sample_id"].astype(str).str.len() == 0).sum())
    duplicate_mask = df["sample_id"].duplicated(keep=False)
    duplicate_sample_id_count = int(df.loc[duplicate_mask, "sample_id"].nunique())
    duplicate_row_count = int(duplicate_mask.sum())
    invalid_label_count = _count_invalid_labels(df["true_label"])
    score_values = pd.to_numeric(df["predicted_score"], errors="coerce").to_numpy()
    non_finite_score_count = int((~np.isfinite(score_values)).sum())
    split_values = set(df["split"].astype(str).str.strip().str.lower())
    non_validation_split_count = int((~df["split"].astype(str).str.strip().str.lower().isin({"val", "validation"})).sum())

    checks_pass = (
        len(df) == expected_count
        and missing_id_count == 0
        and duplicate_sample_id_count == 0
        and invalid_label_count == 0
        and non_finite_score_count == 0
        and non_validation_split_count == 0
    )

    score_series = pd.Series(score_values)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_validation_samples": int(expected_count),
        "exported_prediction_rows": int(len(df)),
        "unique_sample_ids": int(df["sample_id"].nunique(dropna=True)),
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "duplicate_row_count": duplicate_row_count,
        "missing_id_count": missing_id_count,
        "invalid_label_count": invalid_label_count,
        "non_finite_score_count": non_finite_score_count,
        "non_validation_split_count": non_validation_split_count,
        "class_counts": {str(k): int(v) for k, v in df["true_label"].value_counts().sort_index().items()},
        "machine_type_counts": {str(k): int(v) for k, v in df["machine_type"].value_counts().sort_index().items()},
        "machine_id_counts": {str(k): int(v) for k, v in df["machine_id"].value_counts().sort_index().items()},
        "split_values": sorted(split_values),
        "score_minimum": float(score_series.min()),
        "score_maximum": float(score_series.max()),
        "score_mean": float(score_series.mean()),
        "score_standard_deviation": float(score_series.std(ddof=1)),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": repo_commit(),
        "status": status or ("PASS" if checks_pass else "FAIL"),
    }
    return report


def assert_prediction_export_valid(report: dict[str, Any]) -> None:
    assert report["exported_prediction_rows"] == report["expected_validation_samples"], (
        "Prediction row count does not match validation dataset size"
    )
    assert report["missing_id_count"] == 0, "Prediction export contains missing sample_id values"
    assert report["duplicate_sample_id_count"] == 0, "Prediction export contains duplicate sample_id values"
    assert report["invalid_label_count"] == 0, "Prediction export contains invalid labels"
    assert report["non_finite_score_count"] == 0, "Prediction export contains non-finite anomaly scores"
    assert report["non_validation_split_count"] == 0, "Prediction export contains non-validation rows"


def compare_prediction_exports(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_batch_size: int,
    right_batch_size: int,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    left_sorted = left.sort_values("sample_id").reset_index(drop=True)
    right_sorted = right.sort_values("sample_id").reset_index(drop=True)

    sample_ids_match = left_sorted["sample_id"].equals(right_sorted["sample_id"])
    labels_match = left_sorted["true_label"].equals(right_sorted["true_label"])
    scores_left = left_sorted["predicted_score"].to_numpy(dtype=float)
    scores_right = right_sorted["predicted_score"].to_numpy(dtype=float)
    scores_match = bool(np.allclose(scores_left, scores_right, rtol=tolerance, atol=tolerance))
    max_abs_score_difference = float(np.max(np.abs(scores_left - scores_right))) if len(scores_left) else 0.0
    left_auc = float(roc_auc_score(left_sorted["true_label"], scores_left))
    right_auc = float(roc_auc_score(right_sorted["true_label"], scores_right))
    auc_difference = abs(left_auc - right_auc)

    status = (
        "PASS"
        if sample_ids_match and labels_match and scores_match and auc_difference <= tolerance
        else "FAIL"
    )
    return {
        "left_batch_size": int(left_batch_size),
        "right_batch_size": int(right_batch_size),
        "sample_ids_identical_after_sort": sample_ids_match,
        "labels_identical_after_sort": labels_match,
        "scores_equal_within_tolerance": scores_match,
        "score_tolerance": tolerance,
        "max_absolute_score_difference": max_abs_score_difference,
        "left_roc_auc": left_auc,
        "right_roc_auc": right_auc,
        "roc_auc_difference": float(auc_difference),
        "status": status,
    }


def metrics_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix,
    }


def recompute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("INDEPENDENT METRIC RECOMPUTATION")
    print("=" * 60)

    y_true = df["true_label"].to_numpy(dtype=int)
    y_scores = df["predicted_score"].to_numpy(dtype=float)

    assert len(np.unique(y_true)) == 2, "Labels must contain both classes"
    assert len(y_true) == len(y_scores), "Score and label lengths must match"
    assert np.all(np.isfinite(y_scores)), "Anomaly scores must be finite"

    positive_auc = float(roc_auc_score(y_true, y_scores))
    negative_auc = float(roc_auc_score(y_true, -y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_scores)
    youden_values = tpr_arr - fpr_arr
    youden_threshold = float(thresholds[np.argmax(youden_values)])

    fpr_det, fnr_det, _ = det_curve(y_true, y_scores)
    eer_idx = np.nanargmin(np.abs(fnr_det - fpr_det))
    eer = float((fpr_det[eer_idx] + fnr_det[eer_idx]) / 2.0)

    metrics = {
        "roc_auc": positive_auc,
        "positive_score_roc_auc": positive_auc,
        "negative_score_roc_auc": negative_auc,
        "pr_auc": pr_auc,
        "eer": eer,
        "youden_threshold": youden_threshold,
        "metrics_at_threshold_0_5": metrics_at_threshold(y_true, y_scores, 0.5),
        "metrics_at_youden_threshold": metrics_at_threshold(y_true, y_scores, youden_threshold),
    }

    print(f"ROC-AUC: {positive_auc:.10f}")
    print(f"Negative-score ROC-AUC: {negative_auc:.10f}")
    print(f"PR-AUC: {pr_auc:.10f}")
    print(f"EER: {eer:.10f}")
    print(f"Youden threshold: {youden_threshold:.10f}")
    return metrics


def compute_subgroup_metrics(df: pd.DataFrame) -> dict[str, Any]:
    subgroups: dict[str, Any] = {}
    for column, prefix in [("machine_type", "machine_type"), ("machine_id", "machine_id")]:
        for value in sorted(df[column].astype(str).unique()):
            subset = df[df[column].astype(str) == value]
            if subset["true_label"].nunique() == 2:
                subgroups[f"{prefix}_{value}"] = {
                    "roc_auc": float(roc_auc_score(subset["true_label"], subset["predicted_score"])),
                    "count": int(len(subset)),
                }
            else:
                subgroups[f"{prefix}_{value}"] = {
                    "roc_auc": None,
                    "count": int(len(subset)),
                    "reason": "single class",
                }
    return subgroups


def load_original_corrupted_summary() -> dict[str, Any]:
    if not ORIGINAL_CORRUPTED_EXPORT.exists():
        return {"path": str(ORIGINAL_CORRUPTED_EXPORT), "status": "NOT_FOUND"}

    df = pd.read_csv(ORIGINAL_CORRUPTED_EXPORT)
    y_true = df["true_label"].to_numpy(dtype=int)
    scores = df["predicted_score"].to_numpy(dtype=float)
    duplicate_mask = df["sample_id"].duplicated(keep=False)
    return {
        "path": str(ORIGINAL_CORRUPTED_EXPORT),
        "status": "CORRUPTED_EXPORT_RETAINED_FOR_HISTORY",
        "exported_prediction_rows": int(len(df)),
        "unique_sample_ids": int(df["sample_id"].nunique(dropna=True)),
        "duplicate_sample_id_count": int(df.loc[duplicate_mask, "sample_id"].nunique()),
        "duplicate_row_count": int(duplicate_mask.sum()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "negative_score_roc_auc": float(roc_auc_score(y_true, -scores)),
        "root_cause": "sample_id generated from batch_idx * len(labels), which collides after a short final batch",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    unique_batch_sizes = []
    for batch_size in args.batch_sizes:
        if batch_size not in unique_batch_sizes:
            unique_batch_sizes.append(batch_size)
    if len(unique_batch_sizes) < 2:
        print("FAIL: provide at least two distinct batch sizes")
        return 1

    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    configure_reproducible_inference()
    model, device, checkpoint_hash = load_checkpoint_and_model(checkpoint_path)
    if model is None or device is None or checkpoint_hash is None:
        return 1

    exports: dict[int, pd.DataFrame] = {}
    validation_reports: dict[int, dict[str, Any]] = {}
    expected_count: int | None = None
    for batch_size in unique_batch_sizes:
        df, expected = generate_validation_predictions(model, device, batch_size=batch_size)
        expected_count = expected if expected_count is None else expected_count
        if expected != expected_count:
            print("FAIL: validation dataset size changed across exports")
            return 1

        report = build_export_validation_report(
            df=df,
            expected_count=expected,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
        )
        validation_reports[batch_size] = report
        try:
            assert_prediction_export_valid(report)
        except AssertionError as exc:
            report["status"] = "FAIL"
            save_json(output_dir / "prediction_export_validation.json", report)
            print(f"FAIL: {exc}")
            return 1
        exports[batch_size] = df

    primary_batch_size = 32 if 32 in exports else unique_batch_sizes[-1]
    primary_df = exports[primary_batch_size]
    primary_report = validation_reports[primary_batch_size]

    corrected_predictions_path = output_dir / "validation_predictions_corrected.csv"
    primary_df.to_csv(corrected_predictions_path, index=False)
    save_json(output_dir / "prediction_export_validation.json", primary_report)
    print(f"PASS: corrected predictions saved to {corrected_predictions_path}")
    print(f"PASS: export validation report saved to {output_dir / 'prediction_export_validation.json'}")

    left_batch_size, right_batch_size = unique_batch_sizes[:2]
    determinism = compare_prediction_exports(
        exports[left_batch_size],
        exports[right_batch_size],
        left_batch_size=left_batch_size,
        right_batch_size=right_batch_size,
    )
    save_json(output_dir / "prediction_export_determinism.json", determinism)
    if determinism["status"] != "PASS":
        print("FAIL: prediction export differs materially across batch sizes")
        return 1
    print(f"PASS: determinism report saved to {output_dir / 'prediction_export_determinism.json'}")

    metrics = recompute_metrics(primary_df)
    if metrics["positive_score_roc_auc"] <= metrics["negative_score_roc_auc"]:
        print("FAIL: possible score-direction error")
        return 1

    subgroups = compute_subgroup_metrics(primary_df)
    save_json(output_dir / "independent_metrics_corrected.json", metrics)
    save_json(output_dir / "subgroup_metrics_corrected.json", subgroups)

    audit_report = {
        "experiment_id": EXPERIMENT_ID,
        "audit_status": "EVALUATION EXPORT FIXED - PROMPT 2 PASSED",
        "checkpoint_loading": "SUCCESS",
        "prediction_alignment": "VERIFIED_WITH_STABLE_SAMPLE_IDS",
        "label_correctness": "VERIFIED",
        "score_correctness": "VERIFIED",
        "metric_recomputation": "SUCCESS",
        "score_direction": "CORRECT",
        "original_corrupted_export": load_original_corrupted_summary(),
        "corrected_export": {
            "path": str(corrected_predictions_path),
            "batch_size": primary_batch_size,
            "validation_report_path": str(output_dir / "prediction_export_validation.json"),
            "determinism_report_path": str(output_dir / "prediction_export_determinism.json"),
            "export_validation": primary_report,
            "independent_metrics": metrics,
            "subgroup_metrics": subgroups,
        },
        "cause_of_discrepancy": (
            "The previous audit exporter generated sample_id values from batch_idx * len(labels). "
            "When the final validation batch was shorter, later computed IDs overlapped. "
            "The corrected exporter uses the dataset-supplied normalized manifest relative path as sample_id."
        ),
        "git_commit": repo_commit(),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "evaluation_audit.json", audit_report)
    print(f"PASS: evaluation audit JSON saved to {output_dir / 'evaluation_audit.json'}")
    print("FINAL STATUS: EVALUATION EXPORT FIXED - PROMPT 2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
