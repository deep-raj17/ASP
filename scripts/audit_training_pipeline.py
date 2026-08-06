"""
scripts/audit_training_pipeline.py
────────────────────────────────────────────────────────
Training-pipeline and learning audit for EXP-CHAAD-001.

Usage:
    python scripts/audit_training_pipeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from data.dataset import MIMIIDataset
from models.hybrid_model import HybridAnomalyModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def parse_tensorboard_logs():
    """Extract epoch history from TensorBoard logs."""
    print("=" * 60)
    print("TENSORBOARD LOG ANALYSIS")
    print("=" * 60)
    
    log_dir = Path("logs")
    if not log_dir.exists():
        print(f"✗ Logs directory not found: {log_dir}")
        return None
    
    # Try to parse TensorBoard logs
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("⚠ tensorboard not installed, attempting manual log parsing")
        # Manual parsing fallback
        return parse_manual_logs()
    
    # Find event files
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"✗ No TensorBoard event files found")
        return None
    
    print(f"Found {len(event_files)} event files")
    
    # Parse the most recent event file
    latest_file = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"Parsing: {latest_file}")
    
    ea = event_accumulator.EventAccumulator(str(latest_file))
    ea.Reload()
    
    # Extract available tags
    tags = ea.Tags()
    print(f"Available scalar tags: {tags['scalars']}")
    
    epoch_history = {}
    
    # Extract metrics
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        epoch_history[tag] = [(e.step, e.value) for e in events]
    
    print(f"✓ Extracted {len(epoch_history)} metric series")
    
    return epoch_history


def parse_manual_logs():
    """Manual log parsing fallback."""
    print("Attempting manual log parsing...")
    
    # Check for training log file
    log_file = Path("logs/training.log")
    if log_file.exists():
        print(f"Found training log: {log_file}")
        # Parse log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        # Implementation depends on log format
        return None
    
    print("✗ No parseable logs found")
    return None


def reconstruct_epoch_history():
    """Reconstruct epoch history from available sources."""
    print("\n" + "=" * 60)
    print("EPOCH HISTORY RECONSTRUCTION")
    print("=" * 60)
    
    # Try TensorBoard logs
    epoch_history = parse_tensorboard_logs()
    
    if epoch_history is None:
        print("⚠ Cannot reconstruct full epoch history from logs")
        print("Using final reported metrics only")
        
        # Use reported final metrics
        epoch_history = {
            "final_epoch": 100,
            "final_train_loss": 1.0502,
            "final_val_loss": 3.7270,
            "final_val_roc_auc": 0.5233,
            "final_val_eer": 0.4574,
            "final_accuracy_at_05": 0.401,
            "final_balanced_accuracy": 0.543,
            "final_accuracy_at_youden": 0.540
        }
    
    return epoch_history


def audit_loss_components():
    """Audit loss function components."""
    print("\n" + "=" * 60)
    print("LOSS FUNCTION AUDIT")
    print("=" * 60)
    
    loss_components = {
        "bce_weight": cfg.training.bce_weight,
        "contrastive_weight": cfg.training.contrastive_weight,
        "recon_weight": cfg.training.recon_weight,
        "bce_pos_weight": cfg.training.bce_pos_weight,
        "temperature": cfg.training.temperature
    }
    
    print("Loss component weights:")
    for key, value in loss_components.items():
        print(f"  {key}: {value}")
    
    # Check scale
    total_weight = loss_components["bce_weight"] + loss_components["contrastive_weight"] + loss_components["recon_weight"]
    print(f"\nTotal loss weight: {total_weight}")
    
    # Check if one branch dominates
    max_weight = max(loss_components["bce_weight"], loss_components["contrastive_weight"], loss_components["recon_weight"])
    if max_weight / total_weight > 0.7:
        print(f"⚠ One branch dominates: {max_weight / total_weight:.2%}")
    else:
        print(f"✓ Loss weights are balanced")
    
    # Check numerical scale
    print(f"\nNumerical scale analysis:")
    print(f"  BCE weight: {loss_components['bce_weight']}")
    print(f"  Contrastive weight: {loss_components['contrastive_weight']}")
    print(f"  Reconstruction weight: {loss_components['recon_weight']}")
    
    if loss_components["recon_weight"] < 0.1:
        print(f"⚠ Reconstruction weight is very small ({loss_components['recon_weight']})")
    
    return loss_components


def audit_gradients():
    """Audit gradient flow."""
    print("\n" + "=" * 60)
    print("GRADIENT AUDIT")
    print("=" * 60)
    
    # Load model
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = HybridAnomalyModel(cfg.model).to(device)
    
    # Load checkpoint
    checkpoint_path = "artifacts/EXP-CHAAD-001/checkpoint.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Load a single batch
    train_ds = MIMIIDataset(cfg, split="train")
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    
    # Get one batch
    batch = next(iter(train_loader))
    mel = batch["mel"].to(device)
    labels = batch["label"].to(device)
    
    # Enable gradients
    model.train()
    
    # Forward pass
    outputs = model(mel)
    
    # Compute loss
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["logits"].squeeze(-1), 
        labels.float(),
        pos_weight=torch.tensor(cfg.training.bce_pos_weight).to(device)
    )
    
    total_loss = cfg.training.bce_weight * bce_loss
    
    # Backward pass
    total_loss.backward()
    
    # Audit gradients
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            gradient_stats[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "has_grad": True
            }
        else:
            gradient_stats[name] = {
                "has_grad": False
            }
    
    # Count parameters with/without gradients
    with_grad = sum(1 for s in gradient_stats.values() if s["has_grad"])
    without_grad = sum(1 for s in gradient_stats.values() if not s["has_grad"])
    
    print(f"Parameters with gradients: {with_grad}")
    print(f"Parameters without gradients: {without_grad}")
    
    # Check for zero gradients
    zero_grad_params = [name for name, stats in gradient_stats.items() if stats["has_grad"] and stats["norm"] < 1e-10]
    if zero_grad_params:
        print(f"⚠ {len(zero_grad_params)} parameters have near-zero gradients")
    else:
        print(f"✓ No parameters have near-zero gradients")
    
    # Check for NaN/Inf gradients
    nan_grad_params = [name for name, stats in gradient_stats.items() if stats["has_grad"] and (np.isnan(stats["mean"]) or np.isinf(stats["mean"]))]
    if nan_grad_params:
        print(f"✗ {len(nan_grad_params)} parameters have NaN/Inf gradients")
    else:
        print(f"✓ No parameters have NaN/Inf gradients")
    
    # Sample gradient norms
    print(f"\nSample gradient norms:")
    for i, (name, stats) in enumerate(list(gradient_stats.items())[:10]):
        if stats["has_grad"]:
            print(f"  {name}: norm={stats['norm']:.6f}")
    
    return gradient_stats


def audit_data_during_training():
    """Audit data during training."""
    print("\n" + "=" * 60)
    print("DATA AUDIT DURING TRAINING")
    print("=" * 60)
    
    # Load training data
    train_ds = MIMIIDataset(cfg, split="train")
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    
    # Inspect several batches
    batch_stats = []
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Inspect 5 batches
            break
        
        mel = batch["mel"]
        labels = batch["label"]
        
        stats = {
            "batch_idx": i,
            "tensor_shape": list(mel.shape),
            "value_range": (mel.min().item(), mel.max().item()),
            "mean": mel.mean().item(),
            "std": mel.std().item(),
            "label_distribution": labels.long().bincount().tolist(),
            "has_nan": torch.isnan(mel).any().item(),
            "has_inf": torch.isinf(mel).any().item(),
            "all_zero": (mel == 0).all().item()
        }
        
        batch_stats.append(stats)
    
    print(f"\nBatch statistics:")
    for stats in batch_stats:
        print(f"  Batch {stats['batch_idx']}:")
        print(f"    Shape: {stats['tensor_shape']}")
        print(f"    Range: [{stats['value_range'][0]:.4f}, {stats['value_range'][1]:.4f}]")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Labels: {stats['label_distribution']}")
        print(f"    NaN: {stats['has_nan']}, Inf: {stats['has_inf']}, All zero: {stats['all_zero']}")
    
    # Check for issues
    has_issues = False
    for stats in batch_stats:
        if stats["has_nan"] or stats["has_inf"] or stats["all_zero"]:
            has_issues = True
            break
    
    if has_issues:
        print(f"\n✗ Data issues detected")
    else:
        print(f"\n✓ No data issues detected")
    
    return batch_stats


def tiny_batch_overfit_test():
    """Tiny batch overfit test."""
    print("\n" + "=" * 60)
    print("TINY BATCH OVERFIT TEST")
    print("=" * 60)
    
    print("Creating tiny dataset (16 samples)...")
    
    # Load full training dataset
    train_ds = MIMIIDataset(cfg, split="train")
    
    # Create tiny subset (16 samples)
    tiny_indices = list(range(16))
    tiny_ds = Subset(train_ds, tiny_indices)
    tiny_loader = DataLoader(tiny_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # Create fresh model
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = HybridAnomalyModel(cfg.model).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training loop
    epochs = 50
    losses = []
    
    print(f"Training for {epochs} epochs on 16 samples...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tiny_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(mel)
            
            # Simple BCE loss
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["logits"].squeeze(-1),
                labels.float()
            )
            
            loss = bce_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss:.6f}")
    
    # Check if overfitting occurred
    final_loss = losses[-1]
    initial_loss = losses[0]
    loss_reduction = (initial_loss - final_loss) / initial_loss
    
    print(f"\nOverfit test results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction:.2%}")
    
    if loss_reduction > 0.9:
        print(f"✓ Model successfully overfitted tiny batch")
        classification = "TRAINING CAPABLE"
    elif loss_reduction > 0.5:
        print(f"⚠ Model partially overfitted tiny batch")
        classification = "TRAINING PARTIALLY CAPABLE"
    else:
        print(f"✗ Model failed to overfit tiny batch")
        classification = "TRAINING BROKEN"
    
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "classification": classification,
        "losses": losses
    }


def main():
    print("=" * 60)
    print("TRAINING PIPELINE AND LEARNING AUDIT")
    print("EXP-CHAAD-001")
    print("=" * 60)
    
    # Reconstruct epoch history
    epoch_history = reconstruct_epoch_history()
    
    # Audit loss components
    loss_components = audit_loss_components()
    
    # Audit gradients
    gradient_stats = audit_gradients()
    
    # Audit data during training
    batch_stats = audit_data_during_training()
    
    # Tiny batch overfit test
    overfit_results = tiny_batch_overfit_test()
    
    # Create audit report
    audit_report = {
        "experiment_id": "EXP-CHAAD-001",
        "epoch_history": epoch_history,
        "loss_components": loss_components,
        "gradient_audit": {
            "total_parameters": len(gradient_stats),
            "with_gradients": sum(1 for s in gradient_stats.values() if s["has_grad"]),
            "without_gradients": sum(1 for s in gradient_stats.values() if not s["has_grad"]),
            "zero_gradient_count": sum(1 for s in gradient_stats.values() if s["has_grad"] and s.get("norm", 1) < 1e-10)
        },
        "data_audit": {
            "batches_inspected": len(batch_stats),
            "has_issues": any(s["has_nan"] or s["has_inf"] or s["all_zero"] for s in batch_stats)
        },
        "overfit_test": overfit_results
    }
    
    # Save results
    output_dir = Path("artifacts/EXP-CHAAD-001")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "training_audit.json", 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n✓ Training audit saved to: {output_dir / 'training_audit.json'}")
    
    # Final classification
    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION")
    print("=" * 60)
    
    if overfit_results["classification"] == "TRAINING CAPABLE":
        print("FINAL STATUS: TRAINING PIPELINE VERIFIED")
    elif overfit_results["classification"] == "TRAINING PARTIALLY CAPABLE":
        print("FINAL STATUS: TRAINING PIPELINE PARTIALLY VERIFIED")
    else:
        print("FINAL STATUS: TRAINING BUG CONFIRMED")
    
    print(f"\nOverfit test: {overfit_results['classification']}")
    print(f"Loss reduction: {overfit_results['loss_reduction']:.2%}")


if __name__ == "__main__":
    main()
