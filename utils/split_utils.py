"""Shared manifest-driven split utilities for training and evaluation."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "validation": "validation",
    "valid": "validation",
    "test": "test",
}

REQUIRED_COLUMNS = [
    "file_id",
    "relative_path",
    "absolute_path",
    "noise_condition",
    "machine_type",
    "machine_id",
    "label",
    "split",
    "source_recording",
    "segment_start",
    "segment_end",
    "duration_seconds",
    "sample_rate",
    "num_frames",
    "num_channels",
    "file_size_bytes",
    "sha256",
]


@dataclass
class SplitManifest:
    manifest_path: str
    manifest_checksum: Optional[str]
    split: str
    df: pd.DataFrame
    counts: Dict[str, int]
    machine_ids: Dict[str, List[str]]


def normalize_split_name(split: str) -> str:
    if split is None:
        raise ValueError("Split name is required")
    normalized = str(split).strip().lower()
    if normalized not in ALIASES:
        raise ValueError(f"Unknown split '{split}'. Expected one of: train, validation, test")
    return ALIASES[normalized]


def _read_manifest_checksum(manifest_path: str) -> Optional[str]:
    checksum_path = Path(manifest_path).with_suffix(".sha256")
    if checksum_path.exists():
        return checksum_path.read_text(encoding="utf-8").strip()
    return None


def _compute_manifest_checksum(manifest_path: str) -> str:
    content = Path(manifest_path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def load_manifest_split(
    manifest_path: str,
    split: str,
    expected_checksum: Optional[str] = None,
    validate_integrity: bool = True,
) -> SplitManifest:
    """Load a manifest subset using one authoritative split mapping."""
    manifest_path = str(Path(manifest_path))
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    normalized_target = normalize_split_name(split)
    normalized_splits = df["split"].map(lambda value: ALIASES.get(str(value).strip().lower()))
    if normalized_splits.isna().any():
        invalid = sorted(set(df.loc[normalized_splits.isna(), "split"].astype(str)))
        raise ValueError(f"Manifest contains unknown split values: {invalid}")

    df = df.copy()
    df["_normalized_split"] = normalized_splits

    if validate_integrity:
        expected = expected_checksum or _read_manifest_checksum(manifest_path)
        if expected is not None:
            computed = _compute_manifest_checksum(manifest_path)
            if computed != expected:
                raise ValueError(
                    f"Manifest checksum mismatch for {manifest_path}: expected {expected}, got {computed}"
                )

        split_names = ["train", "validation", "test"]
        split_machine_ids = {
            name: sorted(set(df.loc[df["_normalized_split"] == name, "machine_id"].dropna().astype(str)))
            for name in split_names
        }
        for left_name in split_names:
            for right_name in split_names:
                if left_name >= right_name:
                    continue
                overlap = sorted(set(split_machine_ids[left_name]) & set(split_machine_ids[right_name]))
                if overlap:
                    raise ValueError(
                        f"Machine IDs overlap across splits: {left_name} and {right_name} -> {overlap}"
                    )

        for split_name in split_names:
            if df.loc[df["_normalized_split"] == split_name].empty:
                raise ValueError(f"Manifest split '{split_name}' is empty")

        for required_col in ["machine_id", "source_recording"]:
            if df[required_col].isna().any():
                raise ValueError(f"Manifest contains missing values in column '{required_col}'")

    selected = df.loc[df["_normalized_split"] == normalized_target].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows found for split '{split}'")

    selected = selected.sort_values(["relative_path", "file_id"]).reset_index(drop=True)
    counts = {name: int((df["_normalized_split"] == name).sum()) for name in ["train", "validation", "test"]}
    machine_ids = {
        name: sorted(set(df.loc[df["_normalized_split"] == name, "machine_id"].dropna().astype(str)))
        for name in ["train", "validation", "test"]
    }

    return SplitManifest(
        manifest_path=manifest_path,
        manifest_checksum=_read_manifest_checksum(manifest_path) if expected_checksum is None else expected_checksum,
        split=normalized_target,
        df=selected,
        counts=counts,
        machine_ids=machine_ids,
    )


def get_repo_commit(repo_root: Optional[str] = None) -> Optional[str]:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None
