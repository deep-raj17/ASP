"""
inference/detector.py
────────────────────────────────────────────────────────
Multi-perspective Anomaly Detection Engine.

Anomaly signals computed:
  1. Reconstruction Error  – MSE(input_mel, reconstructed_mel)
  2. Embedding Distance    – Cosine distance to reference mean
  3. Mahalanobis Distance  – via Ledoit-Wolf covariance estimate
  4. Contrastive Score     – cosine similarity to reference pool (k-NN)

Score fusion:
  Weighted sum of the four z-score-normalised, sigmoid-mapped
  signals. Weights from InferenceConfig sum to 1.

Calibration:
  Call fit_reference_distribution() on normal training data.
  This pre-computes μ/σ for each signal so that z-scores are
  meaningful. Without calibration, scores default to 0.5.

Output:
  JSON-compatible dict matching the exact structure in the
  project requirements document.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.covariance import LedoitWolf
from typing import Optional, List

from config import Config


class AnomalyDetector:
    """
    Wraps a trained HybridAnomalyModel for production inference.

    Usage:
        detector = AnomalyDetector(model, cfg)
        detector.fit_reference_distribution(normal_loader)
        result = detector.detect(mel_tensor)
    """

    def __init__(self, model: torch.nn.Module, cfg: Config):
        self.model  = model
        self.cfg    = cfg
        self.icfg   = cfg.inference
        self.device = next(model.parameters()).device
        self.model.eval()

        # Reference distribution (fitted by fit_reference_distribution)
        self.ref_mean:    Optional[np.ndarray] = None
        self.ref_cov_inv: Optional[np.ndarray] = None
        self.ref_pool:    Optional[np.ndarray] = None    # (N, embed_dim)
        self.ref_mean_normed: Optional[np.ndarray] = None
        self.ref_pool_normed: Optional[np.ndarray] = None

        # Per-signal calibration statistics
        self.recon_mu  = 0.0;  self.recon_sigma  = 1.0
        self.embed_mu  = 0.0;  self.embed_sigma  = 1.0
        self.mahal_mu  = 0.0;  self.mahal_sigma  = 1.0
        self.contra_mu = 0.0;  self.contra_sigma = 1.0

    def refresh_reference_cache(self) -> None:
        """Pre-normalize reference vectors used on every inference call."""
        if self.ref_mean is not None:
            self.ref_mean_normed = self.ref_mean / (np.linalg.norm(self.ref_mean) + 1e-8)
        else:
            self.ref_mean_normed = None

        if self.ref_pool is not None:
            pool_norms = np.linalg.norm(self.ref_pool, axis=1, keepdims=True) + 1e-8
            self.ref_pool_normed = self.ref_pool / pool_norms
        else:
            self.ref_pool_normed = None

    # ── Reference Fitting ─────────────────────────────────

    @torch.no_grad()
    def fit_reference_distribution(self, normal_loader):
        """
        Pre-compute reference embeddings and calibration statistics
        from a DataLoader containing ONLY normal samples.
        """
        embed_list: List[np.ndarray] = []
        recon_list: List[np.ndarray] = []

        for batch in normal_loader:
            mel = batch["mel"].to(self.device)
            out = self.model(mel)

            # Per-sample reconstruction error
            recon_err = F.mse_loss(out["reconstruction"], mel, reduction="none")
            recon_err = recon_err.mean(dim=(1, 2, 3))   # (B,)
            recon_list.append(recon_err.cpu().numpy())

            embed_list.append(out["embeddings"].cpu().numpy())

        recon_all  = np.concatenate(recon_list)       # (N,)
        embeds_all = np.vstack(embed_list)            # (N, D)

        self.ref_pool = embeds_all
        self.ref_mean = embeds_all.mean(axis=0)
        self.refresh_reference_cache()

        # Robust covariance estimation (Ledoit-Wolf)
        lw = LedoitWolf()
        lw.fit(embeds_all)
        self.ref_cov_inv = lw.precision_

        # Calibration: compute μ, σ for each signal on the normal set
        self.recon_mu    = float(recon_all.mean())
        self.recon_sigma = float(recon_all.std()) + 1e-8

        # Embedding cosine distances
        norms = np.linalg.norm(embeds_all, axis=1, keepdims=True) + 1e-8
        embeds_normed = embeds_all / norms
        ref_normed    = self.ref_mean_normed
        cos_dists     = 1.0 - embeds_normed @ ref_normed
        self.embed_mu    = float(cos_dists.mean())
        self.embed_sigma = float(cos_dists.std()) + 1e-8

        # Mahalanobis distances
        diff   = embeds_all - self.ref_mean
        mahals = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.ref_cov_inv, diff))
        self.mahal_mu    = float(mahals.mean())
        self.mahal_sigma = float(mahals.std()) + 1e-8

        # Contrastive: 1 – mean(top-k cosine sim) per sample
        sim_matrix = embeds_normed @ embeds_normed.T
        k = min(5, len(embeds_all) - 1)
        if k > 0:
            # Exclude self (diagonal), take top-k
            np.fill_diagonal(sim_matrix, -1.0)
            knn_sims = np.sort(sim_matrix, axis=1)[:, -k:].mean(axis=1)
            contra_dists = 1.0 - knn_sims
        else:
            contra_dists = np.zeros(len(embeds_all))
        self.contra_mu    = float(contra_dists.mean())
        self.contra_sigma = float(contra_dists.std()) + 1e-8

        print(
            f"[Detector] Reference fitted on {len(embeds_all)} normal samples.\n"
            f"  Recon   μ={self.recon_mu:.4f}  σ={self.recon_sigma:.4f}\n"
            f"  EmbDist μ={self.embed_mu:.4f}  σ={self.embed_sigma:.4f}\n"
            f"  Mahal   μ={self.mahal_mu:.4f}  σ={self.mahal_sigma:.4f}\n"
            f"  ConDist μ={self.contra_mu:.4f}  σ={self.contra_sigma:.4f}"
        )

    # ── Single-Sample Inference ───────────────────────────

    @torch.no_grad()
    def detect(self, mel_spec: torch.Tensor) -> dict:
        """
        Run anomaly detection on a single mel spectrogram.

        Args:
            mel_spec : (1, 1, n_mels, frames) tensor

        Returns:
            Structured dict matching the project JSON specification.
        """
        mel_spec = mel_spec.to(self.device)
        out = self.model(mel_spec)

        emb = out["embeddings"].cpu().float().numpy()[0]   # (D,)

        # ── Score 1: Reconstruction Error ─────────────────
        recon_err  = F.mse_loss(out["reconstruction"], mel_spec, reduction="mean").item()
        recon_norm = self._z_score(recon_err, self.recon_mu, self.recon_sigma)

        # ── Score 2: Embedding Distance ───────────────────
        if self.ref_mean is not None:
            emb_n  = emb / (np.linalg.norm(emb) + 1e-8)
            ref_n  = self.ref_mean_normed
            if ref_n is None:
                self.refresh_reference_cache()
                ref_n = self.ref_mean_normed
            embed_dist = float(1.0 - np.dot(emb_n, ref_n))
            embed_norm = self._z_score(embed_dist, self.embed_mu, self.embed_sigma)
        else:
            embed_dist = 0.0
            embed_norm = 0.0

        # ── Score 3: Mahalanobis Distance ─────────────────
        if self.ref_mean is not None and self.ref_cov_inv is not None:
            diff  = emb - self.ref_mean
            mahal = float(np.sqrt(np.maximum(diff @ self.ref_cov_inv @ diff, 0.0)))
            mahal_norm = self._z_score(mahal, self.mahal_mu, self.mahal_sigma)
        else:
            mahal = 0.0
            mahal_norm = 0.0

        # ── Score 4: Contrastive Score ────────────────────
        if self.ref_pool is not None:
            pool_normed = self.ref_pool_normed
            if pool_normed is None:
                self.refresh_reference_cache()
                pool_normed = self.ref_pool_normed
            emb_normed  = emb / (np.linalg.norm(emb) + 1e-8)
            sims = pool_normed @ emb_normed                # (N,)
            k = min(5, len(sims))
            contra_sim  = float(np.sort(sims)[-k:].mean())
            contra_dist = 1.0 - contra_sim
            contra_norm = self._z_score(contra_dist, self.contra_mu, self.contra_sigma)
        else:
            contra_sim  = 1.0
            contra_dist = 0.0
            contra_norm = 0.0

        # ── Fused Anomaly Score ───────────────────────────
        w = self.icfg
        fused = (
            w.w_recon  * self._sigmoid(recon_norm)
            + w.w_embed  * self._sigmoid(embed_norm)
            + w.w_mahal  * self._sigmoid(mahal_norm)
            + w.w_contra * self._sigmoid(contra_norm)
        )
        fused = float(np.clip(fused, 0.0, 1.0))

        # ── Thresholding & Risk ───────────────────────────
        percentile   = float(np.clip(fused * 100, 0, 100))
        is_anomalous = percentile > self.icfg.percentile_threshold

        health = max(0, min(100, int(100 - percentile)))
        if health < w.risk_critical:
            risk = "Critical"
        elif health < w.risk_high:
            risk = "High"
        elif health < w.risk_medium:
            risk = "Medium"
        else:
            risk = "Low"

        recommendation = {
            "Critical": "Stop equipment immediately – critical failure risk detected.",
            "High":     "Schedule urgent inspection within 24 hours.",
            "Medium":   "Monitor closely and plan maintenance this week.",
            "Low":      "No immediate action required. Continue routine monitoring.",
        }[risk]

        # ── Dominant Frequency Bands (from attention) ─────
        attn_raw = out["attention_weights"][0]           # (T, 1)
        attn = attn_raw.squeeze(-1).cpu().float().numpy()     # (T,)
        if attn.ndim == 0:
            attn = np.array([float(attn)])
        n_top = min(5, len(attn))
        top_attn_idx = np.argsort(attn)[-n_top:].tolist()

        # ── Temporal Anomaly (frame-level resolution) ─────
        mel_np   = mel_spec[0, 0].cpu().float().numpy()                 # (n_mels, T_frames)
        recon_np = out["reconstruction"][0, 0].cpu().float().numpy()    # (n_mels, T_frames)
        frame_err = np.mean(np.abs(mel_np - recon_np), axis=0)         # (T_frames,)

        sr  = self.cfg.data.sample_rate
        hop = self.cfg.data.hop_length

        temporal_anomalies = []
        if len(frame_err) > 1:
            frame_threshold = np.percentile(frame_err, 75)
            in_anomaly  = False
            start_frame = 0
            for f_idx in range(len(frame_err)):
                if frame_err[f_idx] > frame_threshold and not in_anomaly:
                    in_anomaly  = True
                    start_frame = f_idx
                elif frame_err[f_idx] <= frame_threshold and in_anomaly:
                    in_anomaly = False
                    sev = float(np.mean(frame_err[start_frame:f_idx]))
                    temporal_anomalies.append({
                        "start":    round(start_frame * hop / sr, 3),
                        "end":      round(f_idx * hop / sr, 3),
                        "severity": round(sev, 4),
                    })
            if in_anomaly:
                sev = float(np.mean(frame_err[start_frame:]))
                temporal_anomalies.append({
                    "start":    round(start_frame * hop / sr, 3),
                    "end":      round(len(frame_err) * hop / sr, 3),
                    "severity": round(sev, 4),
                })

        z_score    = float((fused - 0.5) / 0.15)
        confidence = float(min(abs(fused - 0.5) * 2.0, 1.0))

        return {
            "label":         "Anomalous" if is_anomalous else "Normal",
            "anomaly_score": round(fused, 4),
            "confidence":    round(confidence, 4),
            "multi_scores": {
                "reconstruction_error": round(recon_err,  4),
                "embedding_distance":   round(embed_dist, 4),
                "mahalanobis":          round(mahal,      4),
                "contrastive_score":    round(contra_sim, 4),
            },
            "temporal_anomaly": temporal_anomalies,
            "explainability": {
                "dominant_freq_bands":         top_attn_idx,
                "spectrogram_anomaly_regions": is_anomalous,
            },
            "statistics": {
                "z_score":    round(z_score,    4),
                "percentile": round(percentile, 2),
            },
            "system": {
                "health_index":   health,
                "risk_level":     risk,
                "recommendation": recommendation,
            },
        }

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _z_score(value: float, mu: float, sigma: float) -> float:
        return (value - mu) / (sigma + 1e-8)

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Map z-score → [0, 1] via sigmoid."""
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))))
