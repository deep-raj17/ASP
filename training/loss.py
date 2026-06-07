"""
training/loss.py
────────────────────────────────────────────────────────
Multi-objective loss combining:
  1. BCEWithLogitsLoss      – supervised anomaly classification
  2. SupConLoss (InfoNCE)   – contrastive representation learning
  3. MSE Reconstruction     – autoencoder reconstruction quality

NOTE: Mixup produces soft labels (e.g. 0.37). The BCE loss handles
soft labels natively, but SupCon requires hard 0/1 labels for
positive-pair masking. We round labels to 0/1 for contrastive only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from config import TrainingConfig


# ─────────────────────────────────────────────────────────
#  Supervised Contrastive Loss
# ─────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning (Khosla et al. 2020).

    Pulls same-class embeddings together, pushes different-class
    embeddings apart. Expects L2-normalised embeddings as input.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, D)  L2-normalised feature vectors
            labels     : (B,)    HARD binary labels (0 or 1)
        Returns:
            scalar loss
        """
        device = embeddings.device
        B = embeddings.size(0)

        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix / temperature
        sim = torch.mm(embeddings, embeddings.T) / self.temp   # (B, B)

        # Positive pair mask: same class, excluding self
        labels_col = labels.view(-1, 1)
        pos_mask  = (labels_col == labels_col.T).float()
        self_mask = torch.eye(B, device=device)
        pos_mask  = pos_mask - self_mask       # remove diagonal

        # If no positive pairs exist at all, return zero loss
        if pos_mask.sum() < 0.5:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerically stable log-softmax over non-self entries
        neg_mask = 1.0 - self_mask
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * neg_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean log-probability over positive pairs
        n_pos = pos_mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / n_pos

        loss = -mean_log_prob_pos.mean()
        return loss


# ─────────────────────────────────────────────────────────
#  Multi-Objective Loss
# ─────────────────────────────────────────────────────────

class MultiObjectiveLoss(nn.Module):
    """
    L = α · BCE(logits, labels_soft) + β · SupCon(emb, labels_hard) + γ · MSE(recon, input)

    - BCE uses SOFT labels (supports Mixup)
    - SupCon uses HARD labels (rounded to 0/1)
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.alpha = cfg.bce_weight
        self.beta  = cfg.contrastive_weight
        self.gamma = cfg.recon_weight
        self.bce_pos_weight = cfg.bce_pos_weight

        self.supcon = SupConLoss(temperature=cfg.temperature)
        self.mse    = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs   : dict from model.forward()
            labels    : (B,) float32 targets (may be soft from Mixup)
            mel_input : (B, 1, H, W) original mel for reconstruction loss
        Returns:
            (total_loss, {component_name: value})
        """
        logits     = outputs["logits"].squeeze(-1)       # (B,)
        embeddings = outputs["embeddings"]               # (B, D)
        recon      = outputs["reconstruction"]           # (B, 1, H, W)

        # BCE with soft labels + positive-class reweighting (imbalanced normals vs anomalies)
        pw = torch.tensor(
            self.bce_pos_weight, device=logits.device, dtype=logits.dtype
        )
        loss_bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pw
        )

        # SupCon requires hard 0/1 labels for pair masking
        hard_labels = (labels >= 0.5).float()
        loss_con = self.supcon(embeddings, hard_labels)

        # Reconstruction
        loss_recon = self.mse(recon, mel_input)

        total = (
            self.alpha * loss_bce
            + self.beta  * loss_con
            + self.gamma * loss_recon
        )

        return total, {
            "loss_bce":   loss_bce.item(),
            "loss_con":   loss_con.item(),
            "loss_recon": loss_recon.item(),
            "loss_total": total.item(),
        }
