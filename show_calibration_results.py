#!/usr/bin/env python3
"""
show_calibration_results.py

Create readable reports from checkpoints/detector_calibration.pt.

Usage:
    python show_calibration_results.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import cfg


def describe_value(value: Any) -> dict:
    """Return JSON-friendly metadata for tensors, arrays, and scalar values."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        info = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            finite = value[np.isfinite(value)]
            if finite.size:
                info.update(
                    {
                        "mean": float(finite.mean()),
                        "std": float(finite.std()),
                        "min": float(finite.min()),
                        "max": float(finite.max()),
                    }
                )
        return info

    if isinstance(value, (float, int, str, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}

    return {"type": type(value).__name__, "value": str(value)}


def fmt_float(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (float, int)) else "N/A"


def main() -> None:
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    calib_path = checkpoint_dir / "detector_calibration.pt"
    json_path = checkpoint_dir / "calibration_report.json"
    text_path = checkpoint_dir / "calibration_report.txt"

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path}\n"
            "Run: python calibrate.py --dataset-dir \"E:\\MIMII\""
        )

    calib = torch.load(calib_path, map_location="cpu", weights_only=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(calib_path),
        "calibration_statistics": {
            "reconstruction_error": {
                "mean": calib.get("recon_mu"),
                "std": calib.get("recon_sigma"),
            },
            "embedding_distance": {
                "mean": calib.get("embed_mu"),
                "std": calib.get("embed_sigma"),
            },
            "mahalanobis_distance": {
                "mean": calib.get("mahal_mu"),
                "std": calib.get("mahal_sigma"),
            },
            "contrastive_distance": {
                "mean": calib.get("contra_mu"),
                "std": calib.get("contra_sigma"),
            },
        },
        "stored_objects": {key: describe_value(value) for key, value in calib.items()},
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ref_pool = report["stored_objects"].get("ref_pool", {})
    ref_pool_shape = ref_pool.get("shape", [])
    normal_samples = ref_pool_shape[0] if ref_pool_shape else "N/A"
    embedding_dim = ref_pool_shape[1] if len(ref_pool_shape) > 1 else "N/A"

    lines = [
        "MIMII Calibration Report",
        "=" * 80,
        f"Generated at       : {report['generated_at']}",
        f"Source file        : {calib_path}",
        f"Normal samples     : {normal_samples}",
        f"Embedding dimension: {embedding_dim}",
        "",
        "Calibration Statistics",
        "-" * 80,
        f"Reconstruction error mean : {fmt_float(calib.get('recon_mu'))}",
        f"Reconstruction error std  : {fmt_float(calib.get('recon_sigma'))}",
        f"Embedding distance mean   : {fmt_float(calib.get('embed_mu'))}",
        f"Embedding distance std    : {fmt_float(calib.get('embed_sigma'))}",
        f"Mahalanobis distance mean : {fmt_float(calib.get('mahal_mu'))}",
        f"Mahalanobis distance std  : {fmt_float(calib.get('mahal_sigma'))}",
        f"Contrastive distance mean : {fmt_float(calib.get('contra_mu'))}",
        f"Contrastive distance std  : {fmt_float(calib.get('contra_sigma'))}",
        "",
        "Stored Objects",
        "-" * 80,
    ]

    for key, value in report["stored_objects"].items():
        shape = value.get("shape")
        if shape:
            lines.append(f"{key:16s}: shape={shape}, dtype={value.get('dtype')}")
        else:
            lines.append(f"{key:16s}: {value.get('value')}")

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReadable JSON saved: {json_path}")
    print(f"Readable text saved: {text_path}")


if __name__ == "__main__":
    main()
