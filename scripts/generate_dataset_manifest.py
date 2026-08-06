"""
scripts/generate_dataset_manifest.py
────────────────────────────────────────────────────────
Generate a comprehensive dataset manifest for data leakage audit.

Usage:
    python scripts/generate_dataset_manifest.py --config configs/audit_config.yaml
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml
import soundfile as sf


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_mimii_path(filepath: str, dataset_root: str) -> Dict[str, str]:
    """
    Parse MIMII dataset path to extract metadata.
    
    Expected structure: E:/MIMII/0_db_fan/fan/id_00/normal/00000000.wav
    
    Returns:
        dict with keys: noise_condition, machine_type, machine_id, label
    """
    rel_path = os.path.relpath(filepath, dataset_root)
    parts = Path(rel_path).parts
    
    # Initialize with defaults
    metadata = {
        "noise_condition": "unknown",
        "machine_type": "unknown",
        "machine_id": "unknown",
        "label": "unknown"
    }
    
    # Parse noise condition (e.g., 0_db_fan, -6_dB_pump, 6_db_slider)
    for part in parts:
        part_lower = part.lower()
        if "_db_" in part_lower or part_lower.endswith("_db"):
            tokens = part_lower.split("_")
            try:
                db_idx = next(i for i, t in enumerate(tokens) if t == "db")
                noise = "_".join(tokens[:db_idx + 1])
                noise = noise.replace("_db", "_dB").replace("_Db", "_dB")
                metadata["noise_condition"] = noise
                
                # Machine type is everything after 'db'
                remaining = tokens[db_idx + 1:]
                if remaining:
                    metadata["machine_type"] = "_".join(remaining)
            except StopIteration:
                pass
    
    # Parse machine ID (e.g., id_00, id_02)
    for part in parts:
        part_lower = part.lower()
        if part_lower.startswith("id_"):
            metadata["machine_id"] = part_lower
    
    # Parse label (normal/abnormal)
    for part in parts:
        part_lower = part.lower()
        if part_lower == "normal":
            metadata["label"] = "normal"
        elif part_lower == "abnormal":
            metadata["label"] = "abnormal"
    
    return metadata


def assign_split(filepath: str, dataset_root: str, split_seed: int, val_fraction: float) -> str:
    """Assign a deterministic, machine-independent split for a file path."""
    import hashlib

    machine_id = ""
    for part in Path(filepath).parts:
        part_lower = part.lower()
        if part_lower.startswith("id_"):
            machine_id = part_lower
            break

    if machine_id:
        # Machine-independent protocol: explicit assignment
        # id_00, id_02 -> train
        # id_04 -> val
        # id_06 -> test
        if machine_id in ["id_00", "id_02"]:
            return "train"
        elif machine_id == "id_04":
            return "val"
        elif machine_id == "id_06":
            return "test"
        else:
            # Fallback for unknown machine IDs
            bucket = int(hashlib.md5(machine_id.encode()).hexdigest(), 16) % 3
            if bucket == 0:
                return "train"
            elif bucket == 1:
                return "val"
            else:
                return "test"

    rel = os.path.normpath(os.path.relpath(filepath, dataset_root)).lower()
    key = f"{split_seed}|{rel}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    is_val = (h % 10_000) < int(val_fraction * 10_000)
    return "val" if is_val else "train"


def get_audio_info(filepath: str) -> Dict:
    """Get audio file information using soundfile."""
    try:
        info = sf.info(filepath)
        return {
            "duration_seconds": info.duration,
            "sample_rate": info.samplerate,
            "num_frames": info.frames,
            "num_channels": info.channels
        }
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return {
            "duration_seconds": 0,
            "sample_rate": 0,
            "num_frames": 0,
            "num_channels": 0
        }


def generate_manifest(
    dataset_root: str,
    output_path: str,
    split_seed: int,
    val_fraction: float,
    allowed_machine_types: Optional[List[str]] = None,
    allowed_noise_conditions: Optional[List[str]] = None,
    allowed_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate dataset manifest with all required fields.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Find all wav files
    wav_files = list(dataset_root.glob("**/*.wav"))
    print(f"Found {len(wav_files)} WAV files in dataset")
    
    records = []
    unknown_splits = set()
    unknown_machine_types = set()
    unknown_labels = set()
    
    for wav_file in wav_files:
        filepath = str(wav_file)
        
        # Parse metadata from path
        metadata = parse_mimii_path(filepath, str(dataset_root))
        
        # Filter by allowed values if specified
        if allowed_machine_types and metadata["machine_type"] not in allowed_machine_types:
            continue
        if allowed_noise_conditions and metadata["noise_condition"] not in allowed_noise_conditions:
            continue
        if allowed_labels and metadata["label"] not in allowed_labels:
            continue
        
        # Track unknown values
        if metadata["machine_type"] == "unknown":
            unknown_machine_types.add(filepath)
        if metadata["label"] == "unknown":
            unknown_labels.add(filepath)
        
        # Assign split
        split = assign_split(filepath, str(dataset_root), split_seed, val_fraction)
        
        # Get file info
        file_size = wav_file.stat().st_size
        sha256 = compute_sha256(filepath)
        audio_info = get_audio_info(filepath)
        
        # Create record
        record = {
            "file_id": f"{metadata['machine_type']}_{metadata['machine_id']}_{Path(filepath).stem}",
            "relative_path": os.path.relpath(filepath, str(dataset_root)),
            "absolute_path": filepath,
            "noise_condition": metadata["noise_condition"],
            "machine_type": metadata["machine_type"],
            "machine_id": metadata["machine_id"],
            "label": metadata["label"],
            "split": split,
            "source_recording": Path(filepath).stem,  # For MIMII, each file is a recording
            "segment_start": 0.0,
            "segment_end": audio_info["duration_seconds"],
            "duration_seconds": audio_info["duration_seconds"],
            "sample_rate": audio_info["sample_rate"],
            "num_frames": audio_info["num_frames"],
            "num_channels": audio_info["num_channels"],
            "file_size_bytes": file_size,
            "sha256": sha256
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Print summary
    print(f"\nManifest Summary:")
    print(f"  Total files: {len(df)}")
    print(f"  Split distribution:\n{df['split'].value_counts()}")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    print(f"  Machine type distribution:\n{df['machine_type'].value_counts()}")
    print(f"  Machine ID distribution:\n{df['machine_id'].value_counts()}")
    print(f"  Noise condition distribution:\n{df['noise_condition'].value_counts()}")
    
    if unknown_machine_types:
        print(f"\nWarning: {len(unknown_machine_types)} files with unknown machine type")
    if unknown_labels:
        print(f"Warning: {len(unknown_labels)} files with unknown label")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest for data leakage audit")
    parser.add_argument("--config", default="configs/audit_config.yaml", help="Path to audit config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate manifest
    df = generate_manifest(
        dataset_root=config['dataset_root'],
        output_path=config['manifest_path'],
        split_seed=config['split_seed'],
        val_fraction=config['val_fraction'],
        allowed_machine_types=config.get('allowed_machine_types'),
        allowed_noise_conditions=config.get('allowed_noise_conditions'),
        allowed_labels=config.get('allowed_labels')
    )
    
    # Save manifest
    df.to_csv(config['manifest_path'], index=False)
    print(f"\nManifest saved to: {config['manifest_path']}")
    
    # Compute and save manifest checksum
    manifest_content = df.to_csv(index=False).encode('utf-8')
    manifest_checksum = hashlib.sha256(manifest_content).hexdigest()
    
    checksum_file = output_dir / "dataset_manifest.sha256"
    with open(checksum_file, 'w') as f:
        f.write(manifest_checksum)
    print(f"Manifest checksum saved to: {checksum_file}")


if __name__ == "__main__":
    main()
