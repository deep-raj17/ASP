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

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import Config
from utils.audio_utils import AudioProcessor, pad_or_trim


class MIMIIDataset(Dataset):
    """
    Returns dicts with keys:
        mel        (1, n_mels, T_frames)  log-mel spectrogram
        mfcc       (1, n_mfcc, T_frames)  MFCC
        waveform   (1, T_samples)
        label      float32   0=normal, 1=abnormal
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
        root = Path(self.dcfg.dataset_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Dataset not found: {root.resolve()}\n"
                "→ Open config.py and set cfg.data.dataset_dir to your MIMII path.\n"
            )

        all_files = glob.glob(str(root / "**" / "*.wav"), recursive=True)
        if not all_files:
            raise RuntimeError(
                f"\n[ERROR] No .wav files found under: {root.resolve()}\n"
                "→ Check that your folder structure matches:\n"
                "  MIMII_DATASET/0_dB_fan/fan/id_00/normal/*.wav\n"
            )

        for fp in all_files:
            fp_lower = fp.lower().replace("\\", "/")

            # Determine label from path containing 'abnormal' or 'normal'
            if "/abnormal/" in fp_lower or "\\abnormal\\" in fp_lower:
                label = 1
            elif "/normal/" in fp_lower or "\\normal\\" in fp_lower:
                label = 0
            else:
                # Skip files not in normal/abnormal folders
                continue

            # Parse machine type and SNR from path
            # Handles: 0_db_fan, 0_dB_fan, 6_db_pump, -6_db_slider, etc.
            machine    = "unknown"
            machine_id = "id_00"
            snr        = "0_dB"

            for part in Path(fp).parts:
                part_lower = part.lower()

                # Look for SNR pattern: X_db_machine (e.g., 0_db_fan, 6_db_pump)
                if "_db_" in part_lower or part_lower.endswith("_db"):
                    # Handle both: 0_db_fan and 0_db (with machine type separate)
                    tokens = part_lower.split("_")
                    try:
                        # Find the 'db' token
                        db_idx = next(i for i, t in enumerate(tokens) if t == "db")
                        snr = "_".join(tokens[:db_idx + 1])  # e.g., "0_db" or "-6_db"

                        # Convert to standard format: 0_dB (capital B)
                        snr = snr.replace("_db", "_dB").replace("_Db", "_dB")

                        # Machine type is everything after 'db'
                        remaining = tokens[db_idx + 1:]
                        if remaining:
                            machine = "_".join(remaining)  # e.g., "fan", "pump", "slider", "valve"
                    except StopIteration:
                        pass

                # Extract machine ID (e.g., id_00, id_02, id_04, id_06)
                if part_lower.startswith("id_"):
                    machine_id = part_lower

            if machine_types and machine not in machine_types:
                continue
            if snr_levels and snr not in snr_levels:
                continue

            # Deterministic split: stable across runs (Python's hash() is salted per process)
            rel = os.path.relpath(fp, root)
            key = f"{self.dcfg.split_seed}|{os.path.normpath(rel).lower()}"
            h   = int(hashlib.md5(key.encode()).hexdigest(), 16)
            is_val = (h % 10_000) < int(self.dcfg.val_fraction * 10_000)

            if self.split == "train" and is_val:
                continue
            if self.split == "val" and not is_val:
                continue

            self.records.append(dict(
                path=fp, label=label,
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
