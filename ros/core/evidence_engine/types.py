from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class EvidenceState(str, Enum):
    DECLARED = "DECLARED"
    COLLECTED = "COLLECTED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    VERIFIED = "VERIFIED"
    FAILED_VERIFICATION = "FAILED_VERIFICATION"
    INCOMPLETE = "INCOMPLETE"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    QUARANTINED = "QUARANTINED"


class EvidenceType(str, Enum):
    FILE_ARTIFACT = "FILE_ARTIFACT"
    DATASET_MANIFEST = "DATASET_MANIFEST"
    SOURCE_REVISION = "SOURCE_REVISION"
    CONFIGURATION = "CONFIGURATION"
    EXPERIMENT_EXECUTION = "EXPERIMENT_EXECUTION"
    METRIC_RECORD = "METRIC_RECORD"
    STATISTICAL_RESULT = "STATISTICAL_RESULT"
    ENVIRONMENT_RECORD = "ENVIRONMENT_RECORD"
    MODEL_CHECKPOINT = "MODEL_CHECKPOINT"
    PUBLICATION_ARTIFACT = "PUBLICATION_ARTIFACT"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"
    AUDIT_REPORT = "AUDIT_REPORT"


@dataclass(frozen=True)
class ChecksumRecord:
    algorithm: str
    version: str
    value: str


@dataclass(frozen=True)
class LineageReference:
    evidence_id: str
    relationship: str = "derived_from"


@dataclass(frozen=True)
class EvidenceSource:
    uri: str
    identity: str
    retrieved_at: Optional[str] = None


@dataclass(frozen=True)
class ProvenanceRecord:
    producer: str
    producer_type: str
    source: EvidenceSource
    environment_identity: str
    tool_version: str
    repository_revision: Optional[str] = None
    workflow_reference: Optional[str] = None
    experiment_reference: Optional[str] = None


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: EvidenceType
    state: EvidenceState
    created_at: str
    collected_at: str
    project_reference: str
    provenance: ProvenanceRecord
    content_checksum: ChecksumRecord
    metadata_checksum: ChecksumRecord
    parents: Tuple[LineageReference, ...] = ()
    access_classification: str = "internal"
    retention_policy: str = "indefinite"
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]
    supersedes: Optional[str] = None


@dataclass(frozen=True)
class VerifierDefinition:
    verifier_id: str
    version: str
    supported_types: Tuple[EvidenceType, ...]
    deterministic: bool
    timeout_seconds: float
    network_permitted: bool = False


@dataclass(frozen=True)
class VerificationRequest:
    evidence_id: str
    verifier_id: str
    correlation_id: str
    idempotency_key: str
    dry_run: bool = False


@dataclass(frozen=True)
class VerificationResult:
    execution_id: str
    evidence_id: str
    verifier_id: str
    verifier_version: str
    state: EvidenceState
    passed: bool
    reason: str
    details: Mapping[str, Any]
    checksum: ChecksumRecord
    timestamp: str


@dataclass(frozen=True)
class GateEvidenceEvaluation:
    gate_id: str
    outcome: str
    evidence_references: Tuple[str, ...]
    verification_references: Tuple[str, ...]
    requirement_outcomes: Mapping[str, str]
    policy_references: Tuple[str, ...]
    evaluation_checksum: str
    evaluator_version: str
    timestamp: str
    correlation_id: str
