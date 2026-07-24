from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


REGISTRY_NAMES = {
    "projects", "datasets", "experiments", "models", "artifacts",
    "publications", "reviews", "workflows", "policies", "approvals", "modules",
}


@dataclass(frozen=True)
class RegistryRecord:
    registry: str
    record_id: str
    entity_id: str
    version: str
    schema_version: str
    event_type: str
    author: str
    timestamp: str
    correlation_id: str
    causation_id: str
    status: str
    payload: Mapping[str, Any]
    parent_references: Tuple[str, ...] = ()
    supersedes: Optional[str] = None
    idempotency_key: str = ""


@dataclass(frozen=True)
class IntegrityIssue:
    sequence: int
    code: str
    message: str


@dataclass(frozen=True)
class IntegrityReport:
    valid: bool
    record_count: int
    head_checksum: str
    issues: Tuple[IntegrityIssue, ...]
