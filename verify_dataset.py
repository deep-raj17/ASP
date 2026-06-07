#!/usr/bin/env python3
"""
verify_dataset.py
────────────────────────────────────────────────────────
Verify dataset structure and configuration before training.

Run this before training to ensure:
  - Dataset path is correct
  - Folder structure is valid
  - Files are accessible
  - Configuration matches dataset
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from data.dataset import MIMIIDataset


def check_path_exists(path, description):
    """Check if a path exists."""
    if os.path.exists(path):
        print(f"  ✓ {description}: {path}")
        return True
    else:
        print(f"  ✗ {description}: {path} NOT FOUND")
        return False


def scan_dataset_structure(root_path):
    """Scan and report dataset structure."""
    print(f"\n📁 Scanning dataset structure...")
    print(f"   Root: {root_path}")

    if not os.path.exists(root_path):
        print(f"   ✗ Path does not exist!")
        return None

    stats = defaultdict(lambda: defaultdict(lambda: {"normal": 0, "abnormal": 0}))

    # Walk through directory
    for root, dirs, files in os.walk(root_path):
        wav_files = [f for f in files if f.endswith('.wav')]
        if not wav_files:
            continue

        root_lower = root.lower().replace("\\", "/")

        # Determine machine type, SNR, and ID from path
        machine = "unknown"
        snr = "unknown"
        machine_id = "unknown"
        label_type = "unknown"

        for part in Path(root).parts:
            part_lower = part.lower()

            # SNR and machine type
            if "_db_" in part_lower:
                tokens = part_lower.split("_")
                try:
                    db_idx = next(i for i, t in enumerate(tokens) if t == "db")
                    snr = "_".join(tokens[:db_idx + 1])
                    machine = "_".join(tokens[db_idx + 1:]) if len(tokens) > db_idx + 1 else "unknown"
                except StopIteration:
                    pass

            # Machine ID
            if part_lower.startswith("id_"):
                machine_id = part_lower

            # Normal/Abnormal
            if "normal" in part_lower:
                if "abnormal" in part_lower:
                    label_type = "abnormal"
                else:
                    label_type = "normal"

        if machine != "unknown" and snr != "unknown" and machine_id != "unknown":
            stats[machine][snr][label_type] += len(wav_files)

    return stats


def print_dataset_stats(stats):
    """Print dataset statistics."""
    if not stats:
        print("   No valid data found!")
        return

    print("\n📊 Dataset Statistics:")
    print("-" * 60)

    total_files = 0
    for machine in sorted(stats.keys()):
        print(f"\n🔧 Machine Type: {machine.upper()}")
        for snr in sorted(stats[machine].keys()):
            normal = stats[machine][snr]["normal"]
            abnormal = stats[machine][snr]["abnormal"]
            total = normal + abnormal
            total_files += total
            print(f"   {snr:8s}: normal={normal:4d}, abnormal={abnormal:4d}, total={total:4d}")

    print("-" * 60)
    print(f"📁 Total files: {total_files}")


def test_audio_loading(sample_path):
    """Test if audio files can be loaded."""
    print(f"\n🎵 Testing audio loading...")
    try:
        waveform, sr = torchaudio.load(sample_path)
        print(f"  ✓ Successfully loaded: {sample_path}")
        print(f"    Sample rate: {sr} Hz")
        print(f"    Duration: {waveform.shape[-1] / sr:.2f} seconds")
        print(f"    Channels: {waveform.shape[0]}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load: {sample_path}")
        print(f"    Error: {e}")
        return False


def test_dataset_loader():
    """Test the MIMIIDataset loader."""
    print(f"\n🔍 Testing MIMIIDataset loader...")
    print("-" * 60)

    try:
        # Try to create dataset
        train_ds = MIMIIDataset(cfg, split="train")
        val_ds = MIMIIDataset(cfg, split="val")

        print(f"  ✓ Training dataset: {len(train_ds)} samples")
        print(f"  ✓ Validation dataset: {len(val_ds)} samples")

        # Try to load a sample
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"\n  Sample keys: {list(sample.keys())}")
            print(f"  Mel shape: {sample['mel'].shape}")
            print(f"  Label: {sample['label']}")
            print(f"  Machine: {sample.get('machine', 'N/A')}")
            print(f"  SNR: {sample.get('snr', 'N/A')}")
            print(f"  Machine ID: {sample.get('machine_id', 'N/A')}")

        return True

    except Exception as e:
        print(f"  ✗ Dataset loader failed!")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("🔍 MIMII Dataset Verification")
    print("=" * 60)

    # 1. Check configuration
    print("\n⚙️  Configuration:")
    print(f"  Dataset path: {cfg.data.dataset_dir}")
    print(f"  Machine types: {cfg.data.machine_types or 'All (fan, pump, slider, valve)'}")
    print(f"  SNR levels: {cfg.data.snr_levels or 'All (-6_dB, 0_dB, 6_dB)'}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Num workers: {cfg.training.num_workers}")

    # 2. Check paths
    print("\n📂 Path Checks:")
    dataset_exists = check_path_exists(cfg.data.dataset_dir, "Dataset root")

    if not dataset_exists:
        print("\n" + "=" * 60)
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print("\nDataset path does not exist!")
        print(f"Current path in config.py: {cfg.data.dataset_dir}")
        print("\nPlease update config.py with the correct path:")
        print('  dataset_dir: str = r"E:\\MIMII"  # Your actual path')
        return 1

    # 3. Scan structure
    stats = scan_dataset_structure(cfg.data.dataset_dir)
    print_dataset_stats(stats)

    # 4. Find a sample file for testing
    sample_file = None
    for root, dirs, files in os.walk(cfg.data.dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                sample_file = os.path.join(root, f)
                break
        if sample_file:
            break

    if sample_file:
        test_audio_loading(sample_file)
    else:
        print("\n  ✗ No .wav files found!")

    # 5. Test dataset loader
    print("\n" + "-" * 60)
    loader_ok = test_dataset_loader()

    # Summary
    print("\n" + "=" * 60)
    if loader_ok:
        print("✅ VERIFICATION PASSED")
        print("=" * 60)
        print("\nYour dataset is ready for training!")
        print("Run: python train.py")
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print("\nPlease check:")
        print("  1. Dataset path in config.py is correct")
        print("  2. Folder structure matches expected format")
        print("  3. All .wav files are valid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
