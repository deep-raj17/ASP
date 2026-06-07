"""
calibrate.py – Run AFTER training to fit the anomaly detector
────────────────────────────────────────────────────────────────
This script:
  1. Loads the best trained model (checkpoints/best_model.pt)
  2. Passes all NORMAL training samples through the model
  3. Fits the Mahalanobis, embedding, contrastive reference distributions
  4. Saves calibration state to: checkpoints/detector_calibration.pt

Run once after training completes:
    python calibrate.py

Dataset location (same MIMII tree as training — must be readable on this machine):
  • Default: cfg.data.dataset_dir in config.py
  • Override:  python calibrate.py --dataset-dir "X:\\path\\to\\MIMII"
  • Or set env: MIMII_DATASET_DIR=X:\\path\\to\\MIMII
"""

import argparse
import os
import sys
import torch

from config import cfg
from data.dataset import get_normal_loader
from models.hybrid_model import HybridAnomalyModel
from inference.detector import AnomalyDetector


def _resolve_dataset_dir(cli_path: str | None) -> str:
    """Prefer CLI, then MIMII_DATASET_DIR, then config default."""
    if cli_path:
        return os.path.expandvars(os.path.expanduser(cli_path.strip()))
    env = os.environ.get("MIMII_DATASET_DIR", "").strip()
    if env:
        return os.path.expandvars(os.path.expanduser(env))
    return cfg.data.dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Fit anomaly detector on normal MIMII train split.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        metavar="PATH",
        help="Root folder of the MIMII dataset (overrides config and MIMII_DATASET_DIR).",
    )
    args = parser.parse_args()

    dataset_root = _resolve_dataset_dir(args.dataset_dir)
    cfg.data.dataset_dir = dataset_root

    cfg.make_dirs()

    ckpt_path  = os.path.join(cfg.training.checkpoint_dir, "best_model.pt")
    calib_path = os.path.join(cfg.training.checkpoint_dir, "detector_calibration.pt")

    if not os.path.exists(ckpt_path):
        print(
            f"\n[ERROR] Checkpoint not found: {ckpt_path}\n"
            "  → Run  python train.py  first.\n"
        )
        sys.exit(1)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"\n[Calibrate] Device: {device}")
    print(f"[Calibrate] Dataset root: {os.path.abspath(dataset_root)}")
    print(f"[Calibrate] Loading model from: {ckpt_path}")

    model = HybridAnomalyModel(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("[Calibrate] Building normal-data loader...")
    normal_loader = get_normal_loader(cfg)

    detector = AnomalyDetector(model, cfg)
    print("[Calibrate] Fitting reference distribution (may take a few minutes)...\n")
    detector.fit_reference_distribution(normal_loader)

    # Save calibration parameters
    calib_state = {
        "ref_mean":     detector.ref_mean,
        "ref_cov_inv":  detector.ref_cov_inv,
        "ref_pool":     detector.ref_pool,
        "recon_mu":     detector.recon_mu,
        "recon_sigma":  detector.recon_sigma,
        "embed_mu":     detector.embed_mu,
        "embed_sigma":  detector.embed_sigma,
        "mahal_mu":     detector.mahal_mu,
        "mahal_sigma":  detector.mahal_sigma,
        "contra_mu":    detector.contra_mu,
        "contra_sigma": detector.contra_sigma,
    }
    torch.save(calib_state, calib_path)
    print(f"\n✅  Calibration saved → {calib_path}")
    print("    You can now run:  python evaluate.py  or  python app.py\n")


if __name__ == "__main__":
    main()
