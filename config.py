"""
config.py – Edit ONLY dataset_dir before running
────────────────────────────────────────────────────────
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # ── CHANGE THIS to your local MIMII dataset root ──────
    # Your dataset path: E:\MIMII
    # Structure: E:\MIMII\0_db_fan\fan\id_00\normal\*.wav
    dataset_dir: str = r"E:\MIMII"  # Using raw string for Windows path

    # Set to a list to restrict, e.g. ["fan", "pump"] or leave None for ALL
    # Your dataset has: fan, pump, slider, valve
    machine_types: Optional[List[str]] = None  # None = use all 4 machine types

    # SNR levels in your dataset: -6_dB, 0_dB, 6_dB
    # Set to ["0_dB"] to only use 0dB noise level, or None for all
    snr_levels:    Optional[List[str]] = None  # None = use all 3 SNR levels

    val_fraction: float = 0.15          # 15 % of files go to validation

    # Deterministic train/val assignment (same split across runs & machines)
    split_seed: int = 42

    # ── Audio ─────────────────────────────────────────────
    sample_rate:        int   = 16_000
    audio_duration_sec: float = 10.0    # pad/trim to fixed length

    # ── Feature Extraction ────────────────────────────────
    n_fft:      int   = 2048
    hop_length: int   = 512
    n_mels:     int   = 128
    n_mfcc:     int   = 40
    fmin:       float = 20.0
    fmax:       float = 8_000.0

    # Scale log-mel to [0, 1] so CNN + AE reconstruction match (was raw dB vs sigmoid)
    normalize_mel: bool = True

    # ── Augmentation ─────────────────────────────────────
    augment_train:      bool  = True
    time_mask_param:    int   = 30
    freq_mask_param:    int   = 15
    noise_std:          float = 0.005
    mixup_alpha:        float = 0.4        # 0 = disabled
    time_stretch_range: tuple = (0.9, 1.1)
    pitch_shift_steps:  int   = 2


@dataclass
class ModelConfig:
    # "efficientnet_b0" | "efficientnet_b2" | "efficientnet_b4" | "resnet50"
    backbone: str = "efficientnet_b4"

    # "transformer" | "bilstm"
    temporal_module: str = "transformer"

    transformer_d_model:  int   = 256
    transformer_nhead:    int   = 8
    transformer_layers:   int   = 4
    transformer_dropout:  float = 0.1

    bilstm_hidden: int = 256
    bilstm_layers: int = 2

    embedding_dim:    int = 256
    proj_hidden_dim:  int = 512
    ae_latent_channels: int = 128


@dataclass
class TrainingConfig:
    device:    str  = "cuda"    # falls back to cpu automatically
    # Windows: Set to 0 to avoid multiprocessing issues
    num_workers: int  = 0       # MUST be 0 on Windows
    pin_memory:  bool = True
    prefetch_factor: int = 2
    mixed_precision: bool = True  # FP16/BF16 - 2x faster on GPU
    gradient_accumulation_steps: int = 2  # Reduced for GPU (effective batch=64)
    max_grad_norm: float = 1.0

    # RTX 4070 SUPER has 12.9GB VRAM - can handle batch_size=32
    batch_size:    int   = 32   # Optimal for 12.9GB VRAM
    epochs:        int   = 100
    learning_rate: float = 1e-4  # Safer after fixing scale-dominated recon loss
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 5
    scheduler:     str   = "onecycle"  # Faster convergence than cosine

    # Extra positive-class weight for BCE (~ n_normal / n_abnormal on MIMII)
    bce_pos_weight: float = 5.0

    # Loss weights (recon was inflated vs dB-scale input; keep small vs BCE)
    bce_weight:         float = 1.0
    contrastive_weight: float = 0.3
    recon_weight:       float = 0.05
    temperature:        float = 0.07

    checkpoint_dir: str = "checkpoints"
    resume_from:    Optional[str] = None   # explicit path wins if file exists

    # Use checkpoints/epoch_{NNN}.pt when resume_from is unset (e.g. 87 -> epoch_087.pt)
    resume_from_epoch: Optional[int] = None

    # If no path and no resume_from_epoch, load latest epoch_*.pt
    auto_resume: bool = True

    log_dir:           str  = "logs"
    use_tensorboard:   bool = True
    use_wandb:         bool = False
    wandb_project:     str  = "mimii-anomaly"
    log_every_n_steps: int  = 50


@dataclass
class InferenceConfig:
    percentile_threshold: float = 95.0

    # Score fusion weights (must sum to 1.0)
    w_recon:  float = 0.30
    w_embed:  float = 0.25
    w_mahal:  float = 0.30
    w_contra: float = 0.15

    # Health index thresholds (0-100)
    risk_critical: int = 20
    risk_high:     int = 50
    risk_medium:   int = 80


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    training:  TrainingConfig  = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def make_dirs(self):
        for d in [self.training.checkpoint_dir, self.training.log_dir]:
            os.makedirs(d, exist_ok=True)


# ── Global singleton used by all scripts ──────────────────
cfg = Config()