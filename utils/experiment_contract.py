"""Frozen submission-recovery experiment contracts and test-access guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


PROTOCOL_ID = "CHAAD-SUBMISSION-RECOVERY-V1"
DEFAULT_PROTOCOL_PATH = Path(
    "reports/submission_recovery/phase_2/FROZEN_EXPERIMENT_PROTOCOL.yaml"
)
DEVELOPMENT_PHASES = frozenset(range(1, 8))


def load_frozen_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {protocol_path}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Frozen protocol must contain a YAML mapping")
    validate_frozen_protocol(protocol)
    return protocol


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Unexpected protocol_id: {protocol.get('protocol_id')!r}")
    if protocol.get("status") != "FROZEN":
        raise ValueError("Experiment protocol must have status FROZEN")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen protocol requires at least three unique seeds")
    split_policy = protocol.get("dataset", {}).get("split_policy", {})
    required_splits = {"train", "validation", "protected_test"}
    if set(split_policy) != required_splits:
        raise ValueError(f"Split policy must contain exactly {sorted(required_splits)}")
    machine_sets = [
        set(split_policy[name].get("machine_ids", []))
        for name in ("train", "validation", "protected_test")
    ]
    if any(left & right for i, left in enumerate(machine_sets) for right in machine_sets[i + 1 :]):
        raise ValueError("Frozen protocol machine IDs overlap across splits")
    if protocol.get("dataset", {}).get("test_access_before_phase_8") != "forbidden":
        raise ValueError("Frozen protocol must forbid test access before Phase 8")


def assert_split_access(
    *,
    phase: int,
    split: str,
    authorization_file: str | Path | None = None,
) -> str:
    """Reject protected-test access unless Phase 8 has explicit authorization."""
    normalized = str(split).strip().lower()
    if normalized in {"val", "valid"}:
        normalized = "validation"
    if normalized not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split!r}")
    if not 1 <= int(phase) <= 10:
        raise ValueError("phase must be an integer from 1 through 10")
    if normalized != "test":
        return normalized
    if int(phase) != 8:
        raise PermissionError(
            f"Protected test access is forbidden in Phase {phase}; Phase 8 authorization is required."
        )
    if authorization_file is None:
        raise PermissionError("Phase 8 test access requires an explicit authorization file")
    path = Path(authorization_file)
    if not path.is_file():
        raise PermissionError(f"Phase 8 authorization file not found: {path}")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("protocol_id") != PROTOCOL_ID or authorization.get("authorized") is not True:
        raise PermissionError("Invalid Phase 8 authorization record")
    return normalized


def canonical_json_hash(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    sections = {}
    for name in ("data", "model", "training", "inference"):
        section = getattr(config, name, None)
        if section is None or not is_dataclass(section):
            raise TypeError("Config must be a dataclass, mapping, or CHAAD Config object")
        sections[name] = asdict(section)
    return sections


def write_immutable_run_contract(path: str | Path, contract: Mapping[str, Any]) -> str:
    """Create, never overwrite, a canonical run contract and return its hash."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(contract), indent=2, sort_keys=True, default=str) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_prediction_export(
    frame: pd.DataFrame,
    *,
    expected_ids: set[str],
    expected_split: str = "validation",
) -> dict[str, Any]:
    required = {"sample_id", "true_label", "predicted_score", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction export missing columns: {missing}")
    ids = frame["sample_id"].astype(str)
    duplicate_count = int(ids.duplicated().sum())
    observed_ids = set(ids)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    scores = pd.to_numeric(frame["predicted_score"], errors="coerce")
    non_finite_count = int((~scores.map(lambda value: pd.notna(value) and float("-inf") < value < float("inf"))).sum())
    split_values = sorted(set(frame["split"].astype(str).str.lower().replace({"val": "validation"})))
    report = {
        "rows": int(len(frame)),
        "unique_ids": int(ids.nunique()),
        "duplicate_id_count": duplicate_count,
        "missing_id_count": len(missing_ids),
        "unexpected_id_count": len(unexpected_ids),
        "non_finite_score_count": non_finite_count,
        "split_values": split_values,
        "status": "PASS",
    }
    errors = []
    if duplicate_count:
        errors.append(f"{duplicate_count} duplicate sample IDs")
    if missing_ids:
        errors.append(f"{len(missing_ids)} expected IDs missing")
    if unexpected_ids:
        errors.append(f"{len(unexpected_ids)} unexpected IDs")
    if non_finite_count:
        errors.append(f"{non_finite_count} non-finite scores")
    if split_values != [expected_split]:
        errors.append(f"unexpected splits: {split_values}")
    if errors:
        report["status"] = "FAIL"
        raise ValueError("; ".join(errors))
    return report
