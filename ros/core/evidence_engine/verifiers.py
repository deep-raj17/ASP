from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from .hashing import hash_file, hash_structured
from .types import EvidenceRecord, EvidenceType

Verifier = Callable[[EvidenceRecord], tuple[bool, str, dict[str, Any]]]


def file_exists(record: EvidenceRecord):
    path = Path(record.provenance.source.uri)
    return path.is_file(), "file exists" if path.is_file() else "file missing", {"path": path.name}


def file_checksum(record: EvidenceRecord):
    path = Path(record.provenance.source.uri)
    if not path.is_file():
        return False, "file missing", {}
    actual = hash_file(path)
    ok = actual == record.content_checksum.value
    return ok, "checksum matches" if ok else "checksum mismatch", {"actual": actual}


def structured_schema(record: EvidenceRecord):
    path = Path(record.provenance.source.uri)
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception as exc:
        return False, f"structured parse failed: {type(exc).__name__}", {}
    return isinstance(value, (dict, list)), "structured content parsed", {"canonical_hash": hash_structured(value)}


def repository_revision(record: EvidenceRecord):
    revision = record.provenance.repository_revision
    ok = bool(revision and len(revision) >= 7)
    return ok, "revision recorded" if ok else "revision missing", {"revision": revision}


def metric_structure(record: EvidenceRecord):
    metadata = dict(record.metadata or {})
    ok = "metric" in metadata and "value" in metadata and isinstance(metadata["value"], (int, float))
    return ok, "metric structure valid" if ok else "metric/value missing", {}


def environment_manifest(record: EvidenceRecord):
    metadata = dict(record.metadata or {})
    required = {"python", "platform", "packages"}
    missing = sorted(required - set(metadata))
    return not missing, "environment complete" if not missing else "environment incomplete", {"missing": missing}


BUILTINS: dict[str, tuple[tuple[EvidenceType, ...], Verifier]] = {
    "file-exists": (tuple(EvidenceType), file_exists),
    "file-checksum": (tuple(EvidenceType), file_checksum),
    "structured-schema": (tuple(EvidenceType), structured_schema),
    "repository-revision": ((EvidenceType.SOURCE_REVISION,), repository_revision),
    "metric-structure": ((EvidenceType.METRIC_RECORD,), metric_structure),
    "environment-manifest": ((EvidenceType.ENVIRONMENT_RECORD,), environment_manifest),
}
