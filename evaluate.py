"""
evaluate.py – Full evaluation on validation set
────────────────────────────────────────────────────────
Computes and prints all metrics:
  Accuracy, Precision, Recall, F1
  ROC-AUC, PR-AUC, pAUC (max_fpr=0.1)
  Log Loss, EER, Confusion Matrix

Usage:
    python evaluate.py
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from config import cfg
from data.dataset import MIMIIDataset
from torch.utils.data import DataLoader
from models.hybrid_model import HybridAnomalyModel
from utils.metrics import compute_metrics, load_threshold_metadata, persist_threshold_metadata, select_threshold
from utils.experiment_contract import assert_split_access
from utils.precision import safe_autocast


def _resolve_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if normalized in {"validation", "val", "valid"}:
        return "val"
    if normalized == "test":
        return "test"
    raise ValueError(f"Unsupported evaluation split: {split_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the validation or test split.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "val", "test"],
        help="Evaluation split; 'validation' uses a validation-selected threshold, 'test' uses a frozen threshold from metadata.",
    )
    parser.add_argument(
        "--threshold-path",
        default="artifacts/threshold_metadata.json",
        help="Path to the persisted threshold metadata file.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        help="Submission-recovery phase. Protected test access is allowed only in Phase 8.",
    )
    parser.add_argument(
        "--authorization-file",
        default=None,
        help="Explicit Phase 8 authorization JSON; ignored for non-test splits.",
    )
    args = parser.parse_args()

    eval_split = _resolve_split_name(args.split)
    assert_split_access(
        phase=args.phase,
        split=eval_split,
        authorization_file=args.authorization_file,
    )

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No checkpoint at: {ckpt_path}\n  → Run python train.py first.\n")
        sys.exit(1)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"

    print(f"\n[Evaluate] Device: {device}  AMP: {amp_enabled}")
    print(f"[Evaluate] Loading model: {ckpt_path}")

    model = HybridAnomalyModel(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print(f"[Evaluate] Building {eval_split} loader...")
    eval_ds = MIMIIDataset(cfg, split=eval_split)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
        persistent_workers=cfg.training.num_workers > 0,
    )

    all_labels: list = []
    all_scores: list = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Inference"):
            mel    = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"]

            with safe_autocast(device, enabled=amp_enabled):
                outputs = model(mel)

            scores = torch.sigmoid(outputs["logits"].squeeze(-1))
            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(scores.cpu().float().numpy().tolist())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)

    print(f"\n  Samples evaluated : {len(y_true)}")
    print(f"  Normal            : {int((y_true < 0.5).sum())}")
    print(f"  Abnormal          : {int((y_true >= 0.5).sum())}\n")

    try:
        # Round soft labels to hard for metric computation
        y_true_hard = (y_true >= 0.5).astype(int)
        if eval_split == "test":
            threshold_metadata = load_threshold_metadata(args.threshold_path)
            if threshold_metadata.get("selected_on") != "validation":
                raise ValueError("Final test evaluation requires a validation-selected threshold metadata entry.")
            if bool(threshold_metadata.get("test_data_used")):
                raise ValueError("Threshold metadata reports test data usage; the test evaluation mode is read-only.")
            threshold = float(threshold_metadata["threshold"])
            print(f"[Evaluate] Using frozen threshold from {args.threshold_path}: {threshold:.6f}")
        else:
            threshold = select_threshold(y_true_hard, y_scores, selected_on="validation")
            persist_threshold_metadata(
                threshold=threshold,
                selected_on="validation",
                test_data_used=False,
                output_path=args.threshold_path,
                validation_metric_value=None,
                number_of_validation_samples=int(len(y_true_hard)),
                class_counts={
                    "normal": int((y_true_hard == 0).sum()),
                    "abnormal": int((y_true_hard == 1).sum()),
                },
            )
            print(f"[Evaluate] Persisted validation threshold: {threshold:.6f}")

        metrics = compute_metrics(y_true_hard, y_scores, threshold=threshold)
        print(metrics.pretty())

        report_path = os.path.join(
            cfg.training.checkpoint_dir,
            "eval_report_test.json" if eval_split == "test" else "eval_report.json",
        )
        with open(report_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n✅  Report saved → {report_path}\n")

    except ValueError as e:
        print(f"[WARNING] Could not compute full metrics: {e}")
        print("  This usually means only one class is present in the validation set.")
        print("  Make sure your dataset has both normal and abnormal samples.\n")


if __name__ == "__main__":
    main()
