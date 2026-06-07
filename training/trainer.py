"""
training/trainer.py
────────────────────────────────────────────────────────
Production GPU training loop.

Features:
  - Mixed Precision via torch.cuda.amp (works on PyTorch 1.6+, 2.0+)
  - Gradient accumulation (configurable steps)
  - Cosine annealing LR with linear warm-up OR OneCycleLR
  - Gradient norm clipping
  - Best-model + per-epoch checkpointing with resume
  - TensorBoard + WandB logging (optional)
  - Full validation metrics per epoch
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from utils.gpu_utils import (
    setup_cuda_optimizations, empty_cache, get_memory_stats,
    GPUMonitor, compile_model
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from training.loss import MultiObjectiveLoss
from utils.metrics import compute_metrics, EvalMetrics

# ── Optional loggers ──────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
    ):
        self.cfg  = cfg
        self.tcfg = cfg.training

        self.device = torch.device(
            self.tcfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = MultiObjectiveLoss(self.tcfg)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.tcfg.learning_rate,
            weight_decay=self.tcfg.weight_decay,
        )

        self.scheduler = self._build_scheduler()

        # AMP GradScaler – only enabled on CUDA
        self._amp_enabled = self.tcfg.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(
            device=str(self.device),
            enabled=self._amp_enabled,
            init_scale=2**16,
            growth_interval=2000,
        )

        self.best_val_loss = float("inf")
        self.start_epoch   = 1
        self.global_step   = 0

        # Logging
        self.tb_writer: Optional[SummaryWriter] = None
        if self.tcfg.use_tensorboard and _HAS_TB:
            self.tb_writer = SummaryWriter(log_dir=self.tcfg.log_dir)
        if self.tcfg.use_wandb and _HAS_WANDB:
            wandb.init(project=self.tcfg.wandb_project)

        # Resume from checkpoint
        if self.tcfg.resume_from and os.path.exists(self.tcfg.resume_from):
            self._load_checkpoint(self.tcfg.resume_from)

    # ── LR Scheduler ──────────────────────────────────────

    def _build_scheduler(self):
        acc = self.tcfg.gradient_accumulation_steps

        # Number of OPTIMIZER STEPS per epoch (not micro-batches)
        steps_per_epoch = max(len(self.train_loader) // acc, 1)
        total_steps     = steps_per_epoch * self.tcfg.epochs
        warmup_steps    = steps_per_epoch * self.tcfg.warmup_epochs

        if self.tcfg.scheduler == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.tcfg.learning_rate,
                total_steps=total_steps,
            )

        # Default: linear warmup → cosine decay
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-6,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    # ── Main Training Loop ────────────────────────────────

    def fit(self):
        # Setup CUDA optimizations at start
        setup_cuda_optimizations()

        # Suppress torch.compile errors on Windows (Triton unavailable)
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        # Optionally compile model for PyTorch 2.x (Linux only - requires Triton)
        # Windows: torch.compile() is skipped (Triton not available on Windows)
        import platform
        if hasattr(torch, "compile") and self.device.type == "cuda" and platform.system() != "Windows":
            print("[Trainer] Compiling model with torch.compile()...")
            self.model = compile_model(self.model, mode="default")
        else:
            print("[Trainer] torch.compile() skipped (Windows or CPU mode)")

        print(f"\n{'='*60}")
        print(f"  Device : {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU    : {torch.cuda.get_device_name(0)}")
            print(f"  Memory : {get_memory_stats()['free_gb']:.2f} GB free")
        print(f"  AMP    : {self._amp_enabled}")
        print(f"  Epochs : {self.tcfg.epochs}   Batch : {self.tcfg.batch_size}")
        print(f"  Grad Accum Steps : {self.tcfg.gradient_accumulation_steps}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.tcfg.epochs + 1):
            train_loss        = self._train_epoch(epoch)
            val_loss, metrics = self._validate(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch, is_best, train_loss, val_loss, metrics)

            print(
                f"Ep {epoch:03d}/{self.tcfg.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"AUC={metrics.roc_auc:.4f} | EER={metrics.eer:.4f} "
                f"| Acc@0.5={metrics.accuracy_at_05:.3f} "
                f"| BalAcc={metrics.balanced_accuracy:.3f} "
                f"(Acc@Youden={metrics.accuracy:.3f})"
                + ("  [BEST]" if is_best else "")
            )

            if self.tcfg.use_wandb and _HAS_WANDB:
                wandb.log({
                    "epoch":    epoch,
                    "val_loss": val_loss,
                    **{k: v for k, v in metrics.to_dict().items()
                       if isinstance(v, float)},
                })

        if self.tb_writer:
            self.tb_writer.close()

        print("\n[OK] Training complete.")
        print(f"    Best checkpoint: {self.tcfg.checkpoint_dir}/best_model.pt")

    # ── Train Epoch ───────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        acc_steps  = self.tcfg.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        bar = tqdm(self.train_loader, desc=f"Ep {epoch:03d} [Train]", leave=False)
        for step, batch in enumerate(bar):
            mel    = batch["mel"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(device_type=str(self.device), enabled=self._amp_enabled):
                outputs = self.model(mel)
                loss, loss_dict = self.criterion(outputs, labels, mel)
                loss = loss / acc_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % acc_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss_dict["loss_total"]
            self.global_step += 1

            if self.global_step % self.tcfg.log_every_n_steps == 0:
                bar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
                if self.tb_writer:
                    for k, v in loss_dict.items():
                        self.tb_writer.add_scalar(f"train/{k}", v, self.global_step)
                    self.tb_writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )
                    # Log GPU memory
                    if self.device.type == "cuda":
                        mem_stats = get_memory_stats()
                        for k, v in mem_stats.items():
                            self.tb_writer.add_scalar(f"system/{k}", v, self.global_step)

        return epoch_loss / max(len(self.train_loader), 1)

    # ── Validation ────────────────────────────────────────

    @torch.inference_mode()  # More efficient than no_grad for inference
    def _validate(self, epoch: int) -> Tuple[float, EvalMetrics]:
        self.model.eval()
        epoch_loss  = 0.0
        all_labels: list = []
        all_scores: list = []

        bar = tqdm(self.val_loader, desc=f"Ep {epoch:03d} [Val]", leave=False)
        for batch in bar:
            mel    = batch["mel"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(device_type=str(self.device), enabled=False):
                outputs = self.model(mel)
                _, loss_dict = self.criterion(outputs, labels, mel)

            epoch_loss += loss_dict["loss_total"]

            scores = torch.sigmoid(outputs["logits"].squeeze(-1).float())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().float().numpy().tolist())

        avg_loss = epoch_loss / max(len(self.val_loader), 1)
        y_true   = np.array(all_labels)
        y_scores = np.array(all_scores)
        # Metrics need hard 0/1 labels (val may include rare soft labels from mixup on train only)
        y_true_hard = (y_true >= 0.5).astype(int)

        try:
            metrics = compute_metrics(y_true_hard, y_scores)
        except ValueError:
            metrics = EvalMetrics(
                accuracy=0, precision=0, recall=0, f1=0,
                roc_auc=0, pr_auc=0, p_auc=0, log_loss=0,
                eer=1.0, threshold=0.5, confusion_matrix=[[0]],
                accuracy_at_05=0.0, f1_at_05=0.0, balanced_accuracy=0.0,
            )

        if self.tb_writer:
            self.tb_writer.add_scalar("val/loss",    avg_loss,        epoch)
            self.tb_writer.add_scalar("val/roc_auc", metrics.roc_auc, epoch)
            self.tb_writer.add_scalar("val/eer",     metrics.eer,     epoch)
            self.tb_writer.add_scalar("val/acc_05", metrics.accuracy_at_05, epoch)
            self.tb_writer.add_scalar("val/bal_acc", metrics.balanced_accuracy, epoch)

        return avg_loss, metrics

    # ── Checkpointing ─────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool,
                         train_loss: float = None, val_loss: float = None,
                         metrics: EvalMetrics = None):
        Path(self.tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        state = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "best_val_loss":        self.best_val_loss,
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "metrics":              metrics,
        }
        ckpt = os.path.join(self.tcfg.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(state, ckpt)
        if is_best:
            torch.save(state, os.path.join(self.tcfg.checkpoint_dir, "best_model.pt"))

    def _load_checkpoint(self, path: str):
        print(f"Resuming from checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.best_val_loss = state["best_val_loss"]
        self.start_epoch   = state["epoch"] + 1
        self.global_step   = state["global_step"]
        print(f"  -> Resumed at epoch {state['epoch']}, step {state['global_step']}")
