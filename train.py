"""
train.py – Main training entry point
────────────────────────────────────────────────────────
Usage:
    python train.py

To resume:
    Set cfg.training.resume_from = "checkpoints/epoch_010.pt"
    in config.py, then re-run.
"""

import sys
import os
import glob
import torch

from config import cfg
from data.dataset import get_dataloaders
from models.hybrid_model import HybridAnomalyModel
from training.trainer import Trainer


def main():
    # ── Sanity check dataset path ──────────────────────────
    if not os.path.exists(cfg.data.dataset_dir):
        print(
            f"\n[ERROR] Dataset directory not found: '{cfg.data.dataset_dir}'\n"
            "  -> Open config.py and set:  dataset_dir = '<your MIMII path>'\n"
        )
        sys.exit(1)

    cfg.make_dirs()

    # ── Hardware info ─────────────────────────────────────
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  MIMII Acoustic Anomaly Detection - Training")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}")
        print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
    print(f"  Dataset : {cfg.data.dataset_dir}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"  Batches/epoch - train={len(train_loader)}, val={len(val_loader)}\n")

    # ── Model ─────────────────────────────────────────────
    print(f"Building model ({cfg.model.backbone} + {cfg.model.temporal_module})...")
    model    = HybridAnomalyModel(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}\n")

    # ── Resolve resume: explicit path > resume_from_epoch > latest ──
    def _resolve_resume():
        td = cfg.training
        if td.resume_from:
            if not os.path.exists(td.resume_from):
                print(f"\n[ERROR] resume_from not found: {td.resume_from}\n")
                sys.exit(1)
            print(f"  [RESUME] Checkpoint: {td.resume_from}\n")
            return
        if td.resume_from_epoch is not None:
            ck = os.path.join(td.checkpoint_dir, f"epoch_{td.resume_from_epoch:03d}.pt")
            if not os.path.exists(ck):
                print(f"\n[ERROR] resume_from_epoch={td.resume_from_epoch} but missing:\n  {ck}\n")
                sys.exit(1)
            td.resume_from = ck
            print(f"  [RESUME] Epoch {td.resume_from_epoch}: {ck}\n")
            return
        if td.auto_resume:
            ckpts = sorted(glob.glob(os.path.join(td.checkpoint_dir, "epoch_*.pt")))
            if ckpts:
                td.resume_from = ckpts[-1]
                print(f"  [RESUME] Latest checkpoint: {td.resume_from}\n")

    _resolve_resume()

    # ── Train ─────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
