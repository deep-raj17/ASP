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
import json
import argparse
import torch
from datetime import datetime, timezone
from pathlib import Path

from config import cfg
from data.dataset import get_dataloaders
from models.hybrid_model import HybridAnomalyModel
from training.trainer import Trainer
from utils.seed import set_seed, seed_worker
from utils.split_utils import get_repo_commit
from utils.experiment_contract import (
    assert_split_access,
    load_frozen_protocol,
    serialize_config,
    write_immutable_run_contract,
)


def _write_provenance_start(output_path: str = "artifacts/experiment_provenance.json"):
    """Record experiment provenance before training starts."""
    provenance = {
        "experiment_id": f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_repo_commit(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "config": {
            "backbone": cfg.model.backbone,
            "temporal_module": cfg.model.temporal_module,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "random_seed": cfg.training.random_seed,
            "split_seed": cfg.data.split_seed,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if output_path != "artifacts/experiment_provenance.json" else "w"
    with output.open(mode, encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Train CHAAD or validate an isolated submission run.")
    parser.add_argument("--submission-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--phase", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    provenance_path = "artifacts/experiment_provenance.json"
    if args.submission_run:
        if not args.run_id:
            parser.error("--submission-run requires --run-id")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if any(char not in allowed for char in args.run_id):
            parser.error("--run-id may contain only letters, digits, '-' and '_'")
        assert_split_access(phase=args.phase, split="train")
        assert_split_access(phase=args.phase, split="validation")
        protocol = load_frozen_protocol()
        seed = args.seed if args.seed is not None else int(protocol["seeds"][0])
        if seed not in protocol["seeds"]:
            parser.error(f"seed {seed} is not in frozen seed set {protocol['seeds']}")
        maximum_epochs = int(protocol["training"]["maximum_epochs"])
        epochs = args.epochs if args.epochs is not None else maximum_epochs
        if not 1 <= epochs <= maximum_epochs:
            parser.error(f"--epochs must be between 1 and frozen maximum {maximum_epochs}")
        run_root = Path("artifacts/submission_recovery/runs") / args.run_id
        cfg.training.random_seed = seed
        cfg.training.epochs = epochs
        cfg.training.checkpoint_dir = str(run_root / "checkpoints")
        cfg.training.log_dir = str(run_root / "logs")
        cfg.training.auto_resume = False
        cfg.training.resume_from = None
        cfg.training.resume_from_epoch = None
        contract = {
            "protocol_id": protocol["protocol_id"],
            "phase": args.phase,
            "run_id": args.run_id,
            "seed": seed,
            "epochs": epochs,
            "splits": ["train", "validation"],
            "config": serialize_config(cfg),
            "git_commit": get_repo_commit(),
        }
        write_immutable_run_contract(run_root / "run_contract.json", contract)
        provenance_path = str(run_root / "provenance.json")
        if args.dry_run:
            if not os.path.isdir(cfg.data.dataset_dir):
                raise FileNotFoundError(f"Dataset directory not found: {cfg.data.dataset_dir}")
            print(f"[Dry run] Valid isolated run contract: {run_root / 'run_contract.json'}")
            return
    elif args.dry_run:
        parser.error("--dry-run is supported only with --submission-run")

    # ── Determinism ────────────────────────────────────────
    set_seed(cfg.training.random_seed, deterministic_cudnn=cfg.training.deterministic_cudnn)
    provenance = _write_provenance_start(provenance_path)

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
    print(f"  Seed    : {cfg.training.random_seed}")
    print(f"  Commit  : {provenance['git_commit']}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}")
        print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
    print(f"  Dataset : {cfg.data.dataset_dir}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(cfg)
    # Apply seed worker for DataLoader reproducibility
    train_loader.worker_init_fn = seed_worker
    val_loader.worker_init_fn = seed_worker
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
