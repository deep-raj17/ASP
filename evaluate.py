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
from utils.metrics import compute_metrics


def main():
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

    print("[Evaluate] Building validation loader...")
    val_ds = MIMIIDataset(cfg, split="val")
    val_loader = DataLoader(
        val_ds,
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
        for batch in tqdm(val_loader, desc="Inference"):
            mel    = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"]

            with autocast(device_type=str(device), enabled=amp_enabled):
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
        metrics = compute_metrics(y_true_hard, y_scores)
        print(metrics.pretty())

        report_path = os.path.join(cfg.training.checkpoint_dir, "eval_report.json")
        with open(report_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n✅  Report saved → {report_path}\n")

    except ValueError as e:
        print(f"[WARNING] Could not compute full metrics: {e}")
        print("  This usually means only one class is present in the validation set.")
        print("  Make sure your dataset has both normal and abnormal samples.\n")


if __name__ == "__main__":
    main()
