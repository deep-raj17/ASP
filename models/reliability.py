"""
models/reliability.py
────────────────────────────────────────────────────────
Reliability-Aware Fusion for Condition-Aware Anomaly Scoring.

This module implements the novel contribution described in
docs/NOVELTY_AND_CONTRIBUTIONS.md: a learned gating function
that estimates how much each anomaly score should be trusted
for a given sample, conditioned on the embedding, machine type,
and noise condition.

Mathematical formulation:
    r_k(x) = g_{φ,k}(ψ(x), m(x), c(x))     -- reliability scores
    w_k(x) = softmax(r_k(x)/τ)              -- fusion weights
    S(x)   = Σ w_k(x) · σ(z_k(x))          -- fused anomaly score

where:
    ψ(x)   : embedding vector from the backbone
    m(x)   : machine type one-hot
    c(x)   : noise condition one-hot
    z_k(x) : calibrated z-score for source k
    τ      : temperature for softmax sharpening

Key design decisions:
- The gating network is a small MLP (negligible overhead vs backbone).
- Metadata (machine type, noise condition) is embedded and concatenated
  with the score vector, NOT the full embedding, to keep it lightweight.
- Training uses pairwise ranking loss on validation data ONLY,
  preventing any test leakage.
- The module is disabled at train time (during backbone training),
  and only activated after calibration.

Usage:
    # After training backbone and calibrating:
    gate = ReliabilityGate(
        embedding_dim=256,
        num_scores=4,
        num_machine_types=4,
        num_noise_conditions=3,
        temperature=0.5,
    )
    gate.fit(val_loader, calibration_stats)   # train on validation
    fused_score = gate(scores, embedding, machine_idx, noise_idx)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────
#  Reliability Gate Module
# ─────────────────────────────────────────────────────────

class ReliabilityGate(nn.Module):
    """
    Sample-dependent gating network that estimates reliability
    scores for each anomaly signal source.

    Architecture:
        Input: [score_vector (K), embedding_summary (D'), machine_emb (8), noise_emb (8)]
        Hidden: 2-layer MLP with LayerNorm + GELU
        Output: K reliability logits → softmax weights
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_scores: int = 4,
        num_machine_types: int = 4,
        num_noise_conditions: int = 3,
        hidden_dim: int = 64,
        temperature: float = 0.5,
        embedding_summary_dim: int = 32,
    ):
        super().__init__()
        self.num_scores = num_scores
        self.temperature = temperature

        # Small metadata embeddings
        self.machine_embed = nn.Embedding(num_machine_types + 1, 8)  # +1 for unknown
        self.noise_embed = nn.Embedding(num_noise_conditions + 1, 8)

        # Project embedding to a compact summary
        self.emb_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_summary_dim),
            nn.LayerNorm(embedding_summary_dim),
            nn.GELU(),
        )

        # Input: scores (K) + emb_summary (D') + machine_emb (8) + noise_emb (8)
        input_dim = num_scores + embedding_summary_dim + 8 + 8

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_scores),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.gate_net, self.emb_projector]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        scores: torch.Tensor,           # (B, K)  calibrated z-scores or calibrated anomaly probs
        embedding: torch.Tensor,        # (B, D)  pooled embedding from backbone
        machine_idx: torch.Tensor,      # (B,)    integer index for machine type
        noise_idx: torch.Tensor,        # (B,)    integer index for noise condition
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scores:      (B, K) base anomaly signals (e.g., calibrated z-scores)
            embedding:   (B, D) feature embedding from backbone
            machine_idx: (B,)   integer machine type index
            noise_idx:   (B,)   integer noise condition index

        Returns:
            fused_score: (B,)   reliability-weighted fused anomaly score
            weights:     (B, K) softmax-normalized reliability weights
        """
        B = scores.size(0)
        device = scores.device

        # Embed metadata
        m_emb = self.machine_embed(machine_idx)          # (B, 8)
        n_emb = self.noise_embed(noise_idx)               # (B, 8)

        # Compact embedding summary
        emb_summary = self.emb_projector(embedding)       # (B, D')

        # Concatenate inputs
        gate_input = torch.cat([scores, emb_summary, m_emb, n_emb], dim=-1)  # (B, input_dim)

        # Compute reliability logits
        reliability_logits = self.gate_net(gate_input)    # (B, K)

        # Temperature-scaled softmax weights
        weights = F.softmax(reliability_logits / self.temperature, dim=-1)  # (B, K)

        # Weighted fusion
        fused_score = (weights * scores).sum(dim=-1)      # (B,)

        return fused_score, weights


# ─────────────────────────────────────────────────────────
#  Reliability Gate Trainer (Validation-Stage)
# ─────────────────────────────────────────────────────────

@dataclass
class ReliabilityTrainingResult:
    best_loss: float
    best_epoch: int
    weights_mean: np.ndarray       # (K,) mean weights across validation
    weights_std: np.ndarray        # (K,) std of weights
    per_machine_weights: Dict[str, np.ndarray]  # machine_type → mean weights
    per_noise_weights: Dict[str, np.ndarray]    # noise_condition → mean weights


class ReliabilityGateTrainer:
    """
    Trains the reliability gate on VALIDATION data using a
    pairwise ranking loss. This ensures no test leakage.

    The loss encourages: S(anomalous) > S(normal) for all pairs.
    """

    def __init__(
        self,
        gate: ReliabilityGate,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.5,
    ):
        self.gate = gate
        self.optimizer = torch.optim.AdamW(
            gate.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.temperature = temperature

    def pairwise_ranking_loss(
        self,
        fused_scores: torch.Tensor,   # (B,)
        labels: torch.Tensor,          # (B,)  0=normal, 1=abnormal
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        For all pairs (i,j) where label_i=1 and label_j=0:
            loss = max(0, margin - (score_i - score_j))

        This is a simplified version; full implementation uses
        all valid pairs with mean pooling.
        """
        B = fused_scores.size(0)
        device = fused_scores.device

        # Separate indices
        anom_mask = (labels >= 0.5)
        norm_mask = ~anom_mask

        anom_scores = fused_scores[anom_mask]   # (A,)
        norm_scores = fused_scores[norm_mask]   # (N,)

        if len(anom_scores) == 0 or len(norm_scores) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute all pairwise differences
        # Expand: (A, N) = anom_scores[:, None] - norm_scores[None, :]
        diff = anom_scores.unsqueeze(1) - norm_scores.unsqueeze(0)  # (A, N)

        # Hinge loss: max(0, margin - diff)
        loss = torch.clamp(margin - diff, min=0.0)

        return loss.mean()

    @torch.inference_mode()
    def _evaluate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute validation loss and collect weight statistics."""
        self.gate.eval()
        total_loss = 0.0
        n_batches = 0
        all_weights: List[np.ndarray] = []
        machine_weight_map: Dict[str, List[np.ndarray]] = {}
        noise_weight_map: Dict[str, List[np.ndarray]] = {}

        for batch in val_loader:
            scores = batch["calibrated_scores"].to(self.gate.gate_net[0].weight.device)
            embedding = batch["embedding"].to(self.gate.gate_net[0].weight.device)
            machine_idx = batch["machine_idx"].to(self.gate.gate_net[0].weight.device)
            noise_idx = batch["noise_idx"].to(self.gate.gate_net[0].weight.device)
            labels = batch["label"].to(self.gate.gate_net[0].weight.device)

            fused, weights = self.gate(scores, embedding, machine_idx, noise_idx)
            loss = self.pairwise_ranking_loss(fused, labels)
            total_loss += loss.item()
            n_batches += 1

            all_weights.append(weights.cpu().numpy())

            # Per-machine weight collection
            machines = batch.get("machine", [])
            for i, m in enumerate(machines):
                if m not in machine_weight_map:
                    machine_weight_map[m] = []
                machine_weight_map[m].append(weights[i].cpu().numpy())

            noises = batch.get("snr", [])
            for i, n in enumerate(noises):
                if n not in noise_weight_map:
                    noise_weight_map[n] = []
                noise_weight_map[n].append(weights[i].cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        all_w = np.concatenate(all_weights, axis=0)  # (N_val, K)

        per_machine = {m: np.mean(np.stack(w), axis=0) for m, w in machine_weight_map.items()}
        per_noise = {n: np.mean(np.stack(w), axis=0) for n, w in noise_weight_map.items()}

        return avg_loss, {
            "all_weights": all_w,
            "per_machine": per_machine,
            "per_noise": per_noise,
        }

    def fit(
        self,
        val_loader: DataLoader,
        evaluation_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> ReliabilityTrainingResult:
        """
        Train the reliability gate on validation data.

        IMPORTANT: This MUST only use the validation split.
        The gate is trained AFTER the backbone is frozen and
        calibration statistics are computed from train_normal.

        Args:
            val_loader: DataLoader with gate-training samples. Each batch must
                        contain: calibrated_scores, embedding, machine_idx,
                        noise_idx, label, machine (str), snr (str).
            evaluation_loader: Disjoint inner-validation loader used for early
                               stopping. Defaults to val_loader only for legacy
                               callers; publication runs must provide it.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            verbose: Print progress.

        Returns:
            ReliabilityTrainingResult with best loss and weight statistics.
        """
        device = next(self.gate.parameters()).device
        self.gate.train()
        selection_loader = evaluation_loader or val_loader

        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        bar = tqdm(range(epochs), desc="Training Reliability Gate", disable=not verbose)
        for epoch in bar:
            self.gate.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in val_loader:
                scores = batch["calibrated_scores"].to(device)
                embedding = batch["embedding"].to(device)
                machine_idx = batch["machine_idx"].to(device)
                noise_idx = batch["noise_idx"].to(device)
                labels = batch["label"].to(device)

                self.optimizer.zero_grad()
                fused, _ = self.gate(scores, embedding, machine_idx, noise_idx)
                loss = self.pairwise_ranking_loss(fused, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Evaluate
            val_loss, stats = self._evaluate(selection_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.gate.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            bar.set_postfix({
                "train_loss": f"{avg_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "patience": patience_counter,
            })

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best state
        if best_state is not None:
            self.gate.load_state_dict(best_state)

        # Final weight statistics
        _, final_stats = self._evaluate(selection_loader)

        return ReliabilityTrainingResult(
            best_loss=best_loss,
            best_epoch=best_epoch,
            weights_mean=final_stats["all_weights"].mean(axis=0),
            weights_std=final_stats["all_weights"].std(axis=0),
            per_machine_weights=final_stats["per_machine"],
            per_noise_weights=final_stats["per_noise"],
        )


# ─────────────────────────────────────────────────────────
#  Metadata Mapper (string labels → integer indices)
# ─────────────────────────────────────────────────────────

class MetadataMapper:
    """Maps string metadata (machine type, noise condition) to integer indices."""

    def __init__(self):
        self.machine_to_idx: Dict[str, int] = {}
        self.noise_to_idx: Dict[str, int] = {}
        self._next_machine_idx = 0
        self._next_noise_idx = 0

    def fit(self, machines: List[str], noises: List[str]):
        for m in sorted(set(machines)):
            if m not in self.machine_to_idx:
                self.machine_to_idx[m] = self._next_machine_idx
                self._next_machine_idx += 1
        for n in sorted(set(noises)):
            if n not in self.noise_to_idx:
                self.noise_to_idx[n] = self._next_noise_idx
                self._next_noise_idx += 1
        return self

    def map_machine(self, machine: str) -> int:
        return self.machine_to_idx.get(machine, len(self.machine_to_idx))

    def map_noise(self, noise: str) -> int:
        return self.noise_to_idx.get(noise, len(self.noise_to_idx))

    @property
    def num_machines(self) -> int:
        return len(self.machine_to_idx)

    @property
    def num_noises(self) -> int:
        return len(self.noise_to_idx)

    def to_dict(self) -> dict:
        return {
            "machine_to_idx": self.machine_to_idx,
            "noise_to_idx": self.noise_to_idx,
        }
