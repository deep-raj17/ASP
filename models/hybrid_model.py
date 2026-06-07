"""
models/hybrid_model.py
────────────────────────────────────────────────────────
SOTA Hybrid Acoustic Anomaly Detection Network.

Branches:
  (a) CNN Backbone  – EfficientNet-B4 / B0 / B2 / ResNet-50
  (b) Temporal      – Transformer Encoder OR BiLSTM
  (c) Attention Pool – interpretable soft-attention pooling
  (d) Projection Head – L2-norm embeddings for SupCon loss
  (e) Classifier Head – binary logit for anomaly score
  (f) Autoencoder Branch – reconstruction error signal
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)

from config import ModelConfig


# ─────────────────────────────────────────────────────────
#  Attention Pooling
# ─────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft-attention pooling over (B, T, D) → (B, D)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        raw    = self.W(x)                    # (B, T, 1)
        w      = torch.softmax(raw, dim=1)    # (B, T, 1)
        pooled = (x * w).sum(dim=1)           # (B, D)
        return pooled, w


# ─────────────────────────────────────────────────────────
#  Convolutional Autoencoder Branch
# ─────────────────────────────────────────────────────────

class ConvAutoEncoder(nn.Module):
    """Input/output: (B, 1, H, W) – matches Mel spectrogram shape."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32,  3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(32,  1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        z     = self.encoder(x)
        recon = self.decoder(z)
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
        return recon, z


# ─────────────────────────────────────────────────────────
#  CNN Backbone Builder  (patches first conv → 1 channel)
# ─────────────────────────────────────────────────────────

def _patch_first_conv(layer: nn.Conv2d) -> nn.Conv2d:
    """Replace a Conv2d that expects 3-channel input with 1-channel."""
    return nn.Conv2d(
        1, layer.out_channels, layer.kernel_size,
        stride=layer.stride, padding=layer.padding, bias=False,
    )


def _build_backbone(name: str):
    """
    Returns (feature_extractor, out_channels).
    Supports: efficientnet_b0, efficientnet_b2, efficientnet_b4, resnet50.
    """
    name = name.lower()

    if name == "efficientnet_b0":
        m = tv_models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        m.features[0][0] = _patch_first_conv(m.features[0][0])
        return nn.Sequential(*list(m.children())[:-2]), 1280

    elif name == "efficientnet_b2":
        m = tv_models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        m.features[0][0] = _patch_first_conv(m.features[0][0])
        return nn.Sequential(*list(m.children())[:-2]), 1408

    elif name == "efficientnet_b4":
        m = tv_models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        m.features[0][0] = _patch_first_conv(m.features[0][0])
        return nn.Sequential(*list(m.children())[:-2]), 1792

    elif name == "resnet50":
        m = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
        m.conv1 = _patch_first_conv(m.conv1)
        return nn.Sequential(*list(m.children())[:-2]), 2048

    else:
        raise ValueError(
            f"Unsupported backbone: '{name}'. "
            "Choose from: efficientnet_b0, efficientnet_b2, efficientnet_b4, resnet50"
        )


# ─────────────────────────────────────────────────────────
#  Temporal Modules
# ─────────────────────────────────────────────────────────

class TransformerTemporal(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,    # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BiLSTMTemporal(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden * 2, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(out)


# ─────────────────────────────────────────────────────────
#  Main Hybrid Model
# ─────────────────────────────────────────────────────────

class HybridAnomalyModel(nn.Module):
    """
    Forward input  : mel_spec  (B, 1, n_mels, frames)
    Forward output : dict
        embeddings        (B, embed_dim)   L2-normalised
        logits            (B, 1)
        reconstruction    (B, 1, n_mels, frames)
        attention_weights (B, T, 1)
        pooled_feat       (B, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mcfg = cfg
        d = cfg.transformer_d_model

        # (a) CNN Backbone
        self.backbone, cnn_out_ch = _build_backbone(cfg.backbone)

        # Project CNN feature maps → d_model
        self.cnn_proj = nn.Sequential(
            nn.LayerNorm(cnn_out_ch),
            nn.Linear(cnn_out_ch, d),
        )

        # (b) Temporal Module
        if cfg.temporal_module == "transformer":
            self.temporal = TransformerTemporal(
                d_model=d,
                nhead=cfg.transformer_nhead,
                num_layers=cfg.transformer_layers,
                dropout=cfg.transformer_dropout,
            )
        elif cfg.temporal_module == "bilstm":
            self.temporal = BiLSTMTemporal(
                input_dim=d,
                hidden=cfg.bilstm_hidden,
                num_layers=cfg.bilstm_layers,
            )
        else:
            raise ValueError(f"Unknown temporal_module: '{cfg.temporal_module}'")

        # (c) Attention Pooling
        self.attn_pool = AttentionPool(d)

        # (d) Projection Head (for contrastive loss)
        self.projector = nn.Sequential(
            nn.Linear(d, cfg.proj_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.proj_hidden_dim),
            nn.Linear(cfg.proj_hidden_dim, cfg.embedding_dim),
        )

        # (e) Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d // 2, 1),
        )

        # (f) Autoencoder Branch
        self.autoencoder = ConvAutoEncoder(latent_channels=cfg.ae_latent_channels)

        self._init_weights()

    def _init_weights(self):
        for module_group in [self.cnn_proj, self.projector, self.classifier]:
            for layer in module_group.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, mel_spec: torch.Tensor) -> dict:
        B = mel_spec.size(0)

        # Autoencoder branch (runs in parallel)
        recon, _ = self.autoencoder(mel_spec)

        # CNN spatial features
        feat = self.backbone(mel_spec)            # (B, C, H, W)
        _, C, H, W = feat.shape

        # Flatten H×W → sequence
        seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, T, C)
        seq = self.cnn_proj(seq)                              # (B, T, d)

        # Temporal modelling
        seq = self.temporal(seq)                 # (B, T, d)

        # Attention pooling
        pooled, attn_w = self.attn_pool(seq)     # (B, d), (B, T, 1)

        # Output heads
        embeddings = F.normalize(self.projector(pooled), dim=-1)  # (B, E)
        logits     = self.classifier(pooled)                      # (B, 1)

        return dict(
            embeddings=embeddings,
            logits=logits,
            reconstruction=recon,
            attention_weights=attn_w,
            pooled_feat=pooled,
        )
