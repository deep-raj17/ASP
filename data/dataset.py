"""
data/dataset.py
────────────────────────────────────────────────────────
Streaming MIMII dataset loader.

Expected folder structure (matches photos provided):

  MIMII_DATASET/
  └── 0_dB_fan/
      └── fan/
          ├── id_00/
          │   ├── normal/    00000000.wav ...
          │   └── abnormal/  00000000.wav ...
          ├── id_02/
          └── id_04/
  └── 6_dB_pump/
      └── pump/ ...

No files are pre-loaded into RAM; only paths are stored.
"""

from __future__ import annotations

import os
import glob
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.split_utils import load_manifest_split


def _split_name_for_path(path: str, cfg: Config) -> str:
    """Assign a deterministic, machine-independent split for a file path."""
    machine_id = ""
    for part in Path(path).parts:
        part_lower = part.lower()
        if part_lower.startswith("id_"):
            machine_id = part_lower
            break

    if machine_id:
        bucket = int(hashlib.md5(machine_id.encode()).hexdigest(), 16) % 3
        if bucket == 0:
            return "train"
        if bucket == 1:
            return "val"
        return "test"

    rel = os.path.normpath(os.path.relpath(path, cfg.data.dataset_dir)).lower()
    key = f"{cfg.data.split_seed}|{rel}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    is_val = (h % 10_000) < int(cfg.data.val_fraction * 10_000)
    return "val" if is_val else "train"

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import Config
from utils.audio_utils import AudioProcessor, pad_or_trim


def _clean_manifest_value(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_relative_path(path_value: str) -> str:
    return _clean_manifest_value(path_value).replace("\\", "/")


class MIMIIDataset(Dataset):
    """
    Returns dicts with keys:
        mel        (1, n_mels, T_frames)  log-mel spectrogram
        mfcc       (1, n_mfcc, T_frames)  MFCC
        waveform   (1, T_samples)
        label      float32   0=normal, 1=abnormal
        sample_id  str       stable manifest-derived sample identifier
        file_path  str       absolute source WAV path
        relative_path str    normalized manifest relative WAV path
        split      str       manifest split name
        machine    str
        machine_id str
        snr        str
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        machine_types: Optional[List[str]] = None,
        snr_levels: Optional[List[str]] = None,
    ):
        self.cfg       = cfg
        self.dcfg      = cfg.data
        self.split     = split
        self.augment   = (split == "train") and self.dcfg.augment_train
        self.target_len = int(self.dcfg.sample_rate * self.dcfg.audio_duration_sec)
        self.processor = AudioProcessor(self.dcfg)
        self.records: List[Dict] = []
        self._scan(machine_types, snr_levels)

    # ── Discovery ─────────────────────────────────────────

    def _scan(self, machine_types, snr_levels):
        manifest_path = getattr(self.dcfg, "manifest_path", None) or "metadata/dataset_manifest.csv"
        split_manifest = load_manifest_split(
            manifest_path=manifest_path,
            split=self.split,
            expected_checksum=getattr(self.dcfg, "manifest_checksum", None),
            validate_integrity=True,
        )

        for _, row in split_manifest.df.iterrows():
            fp = row["absolute_path"]
            fp_lower = fp.lower().replace("\\", "/")

            # Determine label from path containing 'abnormal' or 'normal'
            if "/abnormal/" in fp_lower or "\\abnormal\\" in fp_lower:
                label = 1
            elif "/normal/" in fp_lower or "\\normal\\" in fp_lower:
                label = 0
            else:
                continue

            machine = row.get("machine_type", "unknown")
            machine_id = row.get("machine_id", "unknown")
            snr = row.get("noise_condition", "0_dB")
            relative_path = _normalize_relative_path(
                row.get("relative_path", os.path.relpath(fp, self.dcfg.dataset_dir))
            )
            manifest_sample_id = _clean_manifest_value(row.get("sample_id"))
            sample_id = manifest_sample_id or relative_path

            if machine_types and machine not in machine_types:
                continue
            if snr_levels and snr not in snr_levels:
                continue

            self.records.append(dict(
                path=fp, label=label,
                sample_id=sample_id,
                relative_path=relative_path,
                split=row.get("split", self.split),
                source_recording=_clean_manifest_value(row.get("source_recording")) or relative_path,
                machine=machine, machine_id=machine_id, snr=snr,
            ))

        n_normal   = sum(r["label"] == 0 for r in self.records)
        n_abnormal = sum(r["label"] == 1 for r in self.records)
        print(
            f"[{self.split.upper():5s}] {len(self.records):6d} files  "
            f"(normal={n_normal}, abnormal={n_abnormal})"
        )
        if len(self.records) == 0:
            raise RuntimeError(
                f"No records found for split='{self.split}'. "
                "Check machine_types / snr_levels filters in config.py."
            )

    # ── Dataset Interface ─────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec      = self.records[idx]
        waveform = self._load_wav(rec["path"])
        label    = float(rec["label"])

        # Mixup (only during training)
        if self.augment and self.dcfg.mixup_alpha > 0 and random.random() < 0.3:
            partner_rec = self.records[random.randint(0, len(self.records) - 1)]
            partner_wav = self._load_wav(partner_rec["path"])
            lam      = float(np.random.beta(self.dcfg.mixup_alpha, self.dcfg.mixup_alpha))
            waveform = lam * waveform + (1.0 - lam) * partner_wav
            label    = lam * label + (1.0 - lam) * float(partner_rec["label"])

        mel, mfcc, waveform = self.processor(waveform, augment=self.augment)

        return dict(
            mel=mel,
            mfcc=mfcc,
            waveform=waveform,
            label=torch.tensor(label, dtype=torch.float32),
            sample_id=rec["sample_id"],
            file_path=rec["path"],
            relative_path=rec["relative_path"],
            split=rec["split"],
            source_recording=rec["source_recording"],
            machine=rec["machine"],
            machine_id=rec["machine_id"],
            snr=rec["snr"],
        )

    # ── Internal helpers ──────────────────────────────────

    def _load_wav(self, path: str):
        """Load and preprocess waveform."""
        # Use soundfile for loading (avoids torchcodec dependency)
        import soundfile as sf
        waveform_np, sr = sf.read(path, dtype="float32")

        # Convert to torch tensor (soundfile returns numpy)
        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
        else:
            # Multi-channel: average to mono
            waveform = torch.from_numpy(waveform_np).mean(dim=-1).unsqueeze(0)

        # Resample if needed (use torchaudio for resampling)
        if sr != self.dcfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.dcfg.sample_rate)

        waveform = pad_or_trim(waveform, self.target_len)
        return waveform

# ── DataLoader Factory ────────────────────────────────────

def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = MIMIIDataset(cfg, split="train",
                            machine_types=cfg.data.machine_types,
                            snr_levels=cfg.data.snr_levels)
    val_ds   = MIMIIDataset(cfg, split="val",
                            machine_types=cfg.data.machine_types,
                            snr_levels=cfg.data.snr_levels)

    common = dict(
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
        persistent_workers=cfg.training.num_workers > 0,
    )

    # Domain-aware sampling: balance samples by machine ID
    if getattr(cfg.training, 'domain_aware_sampling', False):
        from torch.utils.data import WeightedRandomSampler
        import collections

        # Count samples per machine ID
        machine_ids = [r['machine_id'] for r in train_ds.records]
        machine_counts = collections.Counter(machine_ids)

        # Calculate weights: inversely proportional to count
        weights = [1.0 / machine_counts[mid] for mid in machine_ids]

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            drop_last=True,
            **common,
        )
        print(f"[Domain-Aware Sampling] Enabled. Machine ID distribution: {dict(machine_counts)}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


# ── Normal-only loader (for detector calibration) ─────────

def get_normal_loader(cfg: Config) -> DataLoader:
    """Returns a DataLoader with ONLY normal train samples for reference fitting."""
    full_ds = MIMIIDataset(cfg, split="train")
    normal_records = [r for r in full_ds.records if r["label"] == 0]
    full_ds.records = normal_records   # filter in-place

    return DataLoader(
        full_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
        persistent_workers=cfg.training.num_workers > 0,
    )
