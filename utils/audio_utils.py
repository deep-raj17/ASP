"""
utils/audio_utils.py
────────────────────────────────────────────────────────
On-the-fly audio feature extraction and augmentation.

Features extracted:
  - Log-Mel Spectrogram (n_mels configurable, default 128)
  - MFCC (n_mfcc configurable, default 40)
  - Raw waveform branch (pass-through)

Augmentations (applied only when augment=True):
  - Additive Gaussian noise on waveform
  - SpecAugment: Time masking + Frequency masking on spectrograms
  - Mixup is handled at batch level in dataset.py
"""

import random
import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Tuple

from config import DataConfig


class AudioProcessor(nn.Module):
    """
    Feature extractor module. Accepts a mono waveform tensor
    (1, T) and returns (mel_db, mfcc, waveform).

    Registered as an nn.Module so internal transforms are
    properly handled by .to(device) if needed.
    """

    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
            normalized=True,
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            melkwargs=dict(
                n_fft=cfg.n_fft,
                n_mels=cfg.n_mels,
                hop_length=cfg.hop_length,
                f_min=cfg.fmin,
                f_max=cfg.fmax,
            ),
        )

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

        # SpecAugment transforms
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)

    @torch.no_grad()
    def forward(
        self, waveform: torch.Tensor, augment: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform : (1, T) float32, amplitude-normalised to [-1, 1]
            augment  : if True, apply stochastic augmentations

        Returns:
            mel_db   : (1, n_mels, frames)  Log-Mel Spectrogram
            mfcc     : (1, n_mfcc, frames)  MFCC coefficients
            waveform : (1, T)               possibly noise-augmented waveform
        """
        if augment:
            waveform = self._augment_waveform(waveform)

        mel    = self.mel_transform(waveform)       # (1, n_mels, frames)
        mel_db = self.amplitude_to_db(mel)
        mfcc   = self.mfcc_transform(waveform)      # (1, n_mfcc, frames)

        if augment:
            mel_db = self._spec_augment(mel_db)
            mfcc   = self._spec_augment(mfcc)

        # Map log-mel dB to [0,1] for CNN + sigmoid autoencoder (match recon target scale)
        if self.cfg.normalize_mel:
            mel_out = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
        else:
            mel_out = mel_db

        return mel_out, mfcc, waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with probability 0.5."""
        if random.random() < 0.5:
            noise = torch.randn_like(waveform) * self.cfg.noise_std
            waveform = (waveform + noise).clamp(-1.0, 1.0)
        return waveform

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment (time + frequency masking) with probability 0.5 each."""
        if random.random() < 0.5:
            spec = self.time_mask(spec)
        if random.random() < 0.5:
            spec = self.freq_mask(spec)
        return spec


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad (with zeros) or trim waveform to exactly `target_length` samples."""
    length = waveform.shape[-1]
    if length > target_length:
        return waveform[..., :target_length]
    elif length < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - length))
    return waveform
