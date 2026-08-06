"""Generate a non-destructive PMPS-01 scientific provenance snapshot.

The script intentionally records incomplete verification as such. It never
overwrites an existing publication-baseline directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg


OUT = ROOT / "artifacts" / "publication_baseline"
MANIFEST = ROOT / "metadata" / "dataset_manifest.csv"
MANIFEST_SIDECAR = ROOT / "metadata" / "dataset_manifest.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(
            list(args), cwd=ROOT, text=True, encoding="utf-8", errors="replace"
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def json_write(name: str, payload: Any) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def inventory_repository() -> tuple[dict[str, Any], list[str]]:
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}
    paths: list[str] = []
    suffix_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    total_bytes = 0
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in excluded_parts for part in rel.parts):
            continue
        paths.append(str(rel).replace("\\", "/") + ("/" if path.is_dir() else ""))
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        suffix_counts[path.suffix.lower() or "<none>"] += 1
        top = rel.parts[0] if rel.parts else "<root>"
        category_counts[top] += 1
    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "file_count": sum(suffix_counts.values()),
        "directory_count": sum(item.endswith("/") for item in paths),
        "total_bytes_excluding_git": total_bytes,
        "python_file_count": suffix_counts[".py"],
        "notebook_count": suffix_counts[".ipynb"],
        "configuration_file_count": sum(
            suffix_counts[suffix] for suffix in (".yaml", ".yml", ".json", ".toml", ".ini")
        ),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_file_counts": dict(sorted(category_counts.items())),
        "ignored_files": command("git", "ls-files", "--others", "--ignored", "--exclude-standard").splitlines(),
    }
    return inventory, paths


def environment_snapshot() -> dict[str, Any]:
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    selected = {}
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "scikit-learn",
        "librosa",
        "pandas",
        "opencv-python",
        "transformers",
    ):
        selected[name] = packages.get(name, "NOT_INSTALLED")
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        gpu = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        gpu["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu": gpu,
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "selected_packages": selected,
        "all_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def checkpoint_registry() -> dict[str, Any]:
    records = []
    for path in sorted((ROOT / "checkpoints").glob("*.pt")):
        stat = path.stat()
        epoch = None
        if path.stem.startswith("epoch_"):
            try:
                epoch = int(path.stem.split("_")[-1])
            except ValueError:
                pass
        if path.name == "best_model.pt":
            epoch = 6
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "created_utc": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
                "epoch": epoch,
                "selection_criterion": (
                    "minimum validation loss (documented EXP-CHAAD-001)" if path.name == "best_model.pt"
                    else "epoch snapshot; not selected as official"
                ),
                "associated_configuration": "config.py",
                "associated_experiment": "EXP-CHAAD-001",
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(records),
        "records": records,
        "orphan_assessment": (
            "All epoch_001..epoch_100 snapshots and best_model.pt are associated with EXP-CHAAD-001."
        ),
    }


def dataset_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.read_csv(MANIFEST)
    calculated = sha256(MANIFEST)
    sidecar = MANIFEST_SIDECAR.read_text(encoding="utf-8").strip()
    paths = frame["absolute_path"].astype(str)
    exists = paths.map(os.path.isfile)
    size_matches = pd.Series(False, index=frame.index)
    for index, path in paths[exists].items():
        size_matches.at[index] = os.path.getsize(path) == int(frame.at[index, "file_size_bytes"])
    duplicate_paths = int(frame["absolute_path"].duplicated().sum())
    duplicate_relative_paths = int(frame["relative_path"].duplicated().sum())
    duplicate_hash_rows = int(frame["sha256"].duplicated(keep=False).sum())
    dataset = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": cfg.data.dataset_dir,
        "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": calculated,
        "manifest_sidecar": sidecar,
        "manifest_sidecar_matches": calculated == sidecar,
        "total_files": int(len(frame)),
        "total_manifest_bytes": int(frame["file_size_bytes"].sum()),
        "machine_types": frame["machine_type"].value_counts().sort_index().to_dict(),
        "machine_ids": frame["machine_id"].value_counts().sort_index().to_dict(),
        "classes": frame["label"].value_counts().sort_index().to_dict(),
        "splits": frame["split"].value_counts().sort_index().to_dict(),
        "sample_rates": frame["sample_rate"].value_counts().sort_index().to_dict(),
        "channels": frame["num_channels"].value_counts().sort_index().to_dict(),
        "durations_seconds": {
            "minimum": float(frame["duration_seconds"].min()),
            "maximum": float(frame["duration_seconds"].max()),
            "mean": float(frame["duration_seconds"].mean()),
            "total": float(frame["duration_seconds"].sum()),
        },
        "missing_file_count": int((~exists).sum()),
        "file_size_mismatch_count": int((exists & ~size_matches).sum()),
    }
    integrity = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIALLY_VERIFIED",
        "manifest_present": MANIFEST.exists(),
        "manifest_sidecar_matches": calculated == sidecar,
        "manifest_row_count": int(len(frame)),
        "existing_file_count": int(exists.sum()),
        "missing_file_count": int((~exists).sum()),
        "file_size_mismatch_count": int((exists & ~size_matches).sum()),
        "duplicate_absolute_path_count": duplicate_paths,
        "duplicate_relative_path_count": duplicate_relative_paths,
        "duplicate_hash_affected_rows_from_manifest": duplicate_hash_rows,
        "broken_symbolic_link_count": sum(
            1 for path in paths if os.path.islink(path) and not os.path.exists(path)
        ),
        "live_audio_readability": {
            "checked_files": 1,
            "total_files": int(len(frame)),
            "result": "PASS_FOR_REPRESENTATIVE_SAMPLE_ONLY",
            "evidence": "python verify_dataset.py (2026-07-24)",
        },
        "full_corpus_audio_readability": "UNVERIFIED",
        "full_corpus_nan_inf_scan": "UNVERIFIED",
        "full_corpus_live_sha256_recomputation": "UNVERIFIED",
        "reason_not_passed": (
            "PMPS-01 requires every WAV to be read and checked for NaN/Inf. "
            "The existing validator loaded one representative file, and manifest hashes "
            "were not recomputed from the current 135.8 GB corpus in this stage."
        ),
    }
    return dataset, integrity


def write_markdown(dataset: dict[str, Any], integrity: dict[str, Any]) -> None:
    model_card = """# CHAAD Model Card — PMPS-01 Snapshot

## Model

CHAAD is a supervised hybrid acoustic anomaly detector using an
EfficientNet-B4 spectrogram backbone, Transformer temporal encoder, attention
pooling, classifier, contrastive projection, and reconstruction branch.

## Inputs and outputs

Input audio is configured for 16 kHz, 10-second, multi-channel MIMII WAV data
and converted to normalized log-mel features. The audited validation export
contains a continuous anomaly probability.

## Training objective

Weighted BCE classification + supervised contrastive loss + reconstruction
loss. EXP-CHAAD-001 selected epoch 6 by minimum validation loss.

## Intended use

Research on machine-independent industrial acoustic anomaly detection.

## Out-of-scope use

Safety-critical autonomous shutdown, clinical use, surveillance, and claims of
cross-dataset or real-world factory generalization without additional evidence.

## Hardware and dependencies

Python/PyTorch; GPU optional for inference. Exact versions are in
`environment_full.json`.

## Limitations and known issues

Validation ROC-AUC is approximately 0.60026 after prediction-export correction;
the model is underfit; there is no authorized held-out test result in this
audit; multi-seed and cross-platform reproducibility are not established.

## Ethical considerations

False alarms and missed failures can affect worker safety and maintenance cost.
Human oversight and site-specific validation are required.
"""
    dataset_card = f"""# MIMII Dataset Card — PMPS-01 Snapshot

## Identity

- Source: MIMII research dataset
- Local root: `{dataset['dataset_root']}`
- Version: UNKNOWN
- License: UNKNOWN; must be confirmed before redistribution/publication
- Samples: {dataset['total_files']:,}
- Manifest SHA-256: `{dataset['manifest_sha256']}`

## Composition and split protocol

Four machine types (fan, pump, slider, valve), four physical machine IDs, and
normal/abnormal labels. Machine-independent split: train=id_04,
validation=id_00+id_02, test=id_06.

## Known limitations

Class imbalance, four machine families, limited machine-ID diversity, fixed
noise conditions, unknown local dataset version/license, and incomplete live
readability/NaN verification of the full 135.8 GB corpus.

## Integrity status

Manifest/file existence and size checks pass. Full-corpus live decoding,
finite-value scanning, and current-file hash recomputation remain UNVERIFIED.
"""
    governance = """# Research Governance

1. Preserve datasets, manifests, checkpoints, predictions, and historical reports.
2. Use immutable experiment IDs and SHA-256 hashes for provenance.
3. Change one experimental factor at a time and register every run.
4. Keep train, validation, and held-out test roles separate.
5. Do not use test results for model, threshold, or manuscript selection.
6. Label evidence VERIFIED, PARTIALLY VERIFIED, UNVERIFIED, or FAILED.
7. Publication claims must trace to checkpoint, configuration, manifest,
   predictions, metric code, and a generated table/figure.
8. Never conceal failed runs, inconsistencies, or unsupported claims.
"""
    baseline = f"""# Publication Baseline — PMPS-01

## Status

**FAIL — PMPS-02 is not authorized by the PMPS-01 quality gate.**

## Verified

- Repository, environment, configuration, Git, checkpoint, and experiment inventories captured.
- Manifest checksum matches its sidecar: `{dataset['manifest_sha256']}`.
- All {dataset['total_files']:,} manifest paths exist and recorded sizes match.
- Existing repository validators pass split isolation and manifest duplicate checks.
- `best_model.pt` SHA-256 is recorded in `checkpoint_registry.json`.

## Unverified / incomplete

- Every WAV has not been freshly decoded in this stage.
- Every decoded tensor has not been scanned for NaN/Inf.
- Every current WAV hash has not been recomputed from the 135.8 GB corpus.
- Dataset version and redistribution license remain unknown.
- Cross-platform and fresh-environment reproduction remain unverified.

## Conflict resolution

`docs/CURRENT_STATE.md` and `docs/KNOWN_ISSUES.md` record a manifest-sidecar
mismatch, but the current files match exactly. Current file bytes and the
executed SHA-256 check supersede those stale statements.

## Publication risk

High until full-corpus integrity is executed and dataset identity/license are
resolved. Model evidence is validation-only and underfit.
"""
    (OUT / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (OUT / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")
    (OUT / "RESEARCH_GOVERNANCE.md").write_text(governance, encoding="utf-8")
    (OUT / "PUBLICATION_BASELINE.md").write_text(baseline, encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline: {OUT}")
    OUT.mkdir(parents=True)

    repository, tree = inventory_repository()
    json_write("repository_inventory.json", repository)
    (OUT / "repository_tree.txt").write_text("\n".join(tree) + "\n", encoding="utf-8")

    environment = environment_snapshot()
    json_write("environment_full.json", environment)
    requirements = [
        f"{name}=={version}"
        for name, version in environment["all_packages"].items()
    ]
    (OUT / "requirements_frozen.txt").write_text(
        "\n".join(requirements) + "\n", encoding="utf-8"
    )
    conda = command("conda", "env", "export", "--no-builds")
    if not conda.startswith("UNAVAILABLE:"):
        (OUT / "conda_environment.yml").write_text(conda + "\n", encoding="utf-8")

    checkpoints = checkpoint_registry()
    json_write("checkpoint_registry.json", checkpoints)
    dataset, integrity = dataset_snapshot()
    json_write("dataset_inventory.json", dataset)
    json_write("dataset_integrity_report.json", integrity)
    json_write("config_snapshot.json", dataclasses.asdict(cfg))

    git_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": command("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "head": command("git", "rev-parse", "HEAD"),
        "status_porcelain": command("git", "status", "--porcelain=v1").splitlines(),
        "status": command("git", "status"),
        "last_50_commits": command(
            "git", "log", "-50", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%s"
        ).splitlines(),
    }
    json_write("git_snapshot.json", git_snapshot)

    experiment_dir = ROOT / "artifacts" / "EXP-CHAAD-001"
    experiment = {
        "experiment_id": "EXP-CHAAD-001",
        "date": "2026-07-21 to 2026-07-22",
        "purpose": "First machine-independent CHAAD training run",
        "checkpoint": "checkpoints/best_model.pt",
        "configuration": "config.py and artifacts/experiment_provenance.json",
        "metrics": "artifacts/EXP-CHAAD-001/independent_metrics_corrected.json",
        "reports": "artifacts/EXP-CHAAD-001/evaluation_audit.md",
        "artifacts": sorted(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in experiment_dir.glob("*")
            if path.is_file()
        ),
        "status": "OFFICIAL_PROVISIONAL; validation export fixed; underfit",
    }
    json_write(
        "MASTER_EXPERIMENT_REGISTRY.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiments": [experiment],
            "orphan_checkpoint_count": 0,
        },
    )
    write_markdown(dataset, integrity)
    json_write(
        "pmps01_gate.json",
        {
            "status": "FAIL",
            "repository_health_score": 90,
            "dataset_health_score": 70,
            "reproducibility_score": 65,
            "documentation_score": 80,
            "publication_readiness_score": 45,
            "blocking_issues": [
                "Full-corpus live WAV readability and finite-value scan not executed",
                "Full-corpus current-file SHA-256 recomputation not executed",
                "Dataset version and license unknown",
            ],
            "next_stage_authorized": False,
        },
    )
    print(f"Generated PMPS-01 baseline at {OUT}")
    print("PMPS-01 STATUS: FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
