"""Immutable workflow domain types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class GateState(str, Enum):
    UNEVALUATED = "UNEVALUATED"
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    BLOCKED = "BLOCKED"
    WAIVED = "WAIVED"


class EvaluationResult(str, Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    BLOCKED = "BLOCKED"
    WAIVED = "WAIVED"


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, order=True)
class WorkflowVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "WorkflowVersion":
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid semantic version: {value}")
        return cls(*(int(part) for part in parts))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class Prerequisite:
    gate_id: str
    accepted_states: Tuple[GateState, ...] = (
        GateState.SATISFIED,
        GateState.WAIVED,
    )


@dataclass(frozen=True)
class GateDefinition:
    id: str
    title: str
    prerequisites: Tuple[Prerequisite, ...] = ()
    entry: bool = False
    terminal: bool = False
    allow_waiver: bool = False
    waiver_policy: Optional[str] = None
    administrative: bool = False
    parallel_group: Optional[str] = None
    retry_limit: int = 0


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    version: WorkflowVersion
    schema_version: str
    gates: Tuple[GateDefinition, ...]
    source: str
    stop_conditions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GateInstance:
    gate_id: str
    state: GateState = GateState.UNEVALUATED
    attempts: int = 0


@dataclass(frozen=True)
class WorkflowInstance:
    id: str
    definition_id: str
    workflow_version: str
    state: WorkflowState
    gates: Mapping[str, GateInstance]
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["gates"] = {
            key: {
                "gate_id": value.gate_id,
                "state": value.state.value,
                "attempts": value.attempts,
            }
            for key, value in self.gates.items()
        }
        return data


@dataclass(frozen=True)
class GateEvaluationInput:
    gate_id: str
    result: EvaluationResult
    evidence_references: Tuple[str, ...]
    evaluator_identity: str
    evaluator_version: str
    policy_references: Tuple[str, ...]
    timestamp: str
    verification_checksum: str
    correlation_id: str
    approval_references: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionRequest:
    workflow_instance_id: str
    action: str
    actor: str
    actor_type: str
    correlation_id: str
    idempotency_key: str
    expected_revision: int
    dry_run: bool = False
    reason: str = ""
    evaluation: Optional[GateEvaluationInput] = None


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    workflow_instance_id: str
    workflow_version: str
    gate_id: Optional[str]
    previous_state: str
    new_state: str
    actor: str
    actor_type: str
    timestamp: str
    correlation_id: str
    request_checksum: str
    evidence_references: Tuple[str, ...]
    approval_references: Tuple[str, ...]
    reason: str
    outcome: str


@dataclass(frozen=True)
class TransitionResult:
    previous_state: str
    requested_state: str
    final_state: str
    mutated: bool
    reason: str
    error_code: Optional[str]
    audit_event_id: Optional[str]
    timestamp: str
    actor: str
    correlation_id: str
    idempotent: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    error_code: str
    severity: Severity
    location: str
    message: str
    suggested_correction: str


@dataclass(frozen=True)
class ExecutionPlan:
    workflow_instance_id: str
    workflow_state: WorkflowState
    ready_gates: Tuple[str, ...]
    blocked_gates: Tuple[str, ...]
    failed_gates: Tuple[str, ...]
    completed_gates: Tuple[str, ...]
    unmet_prerequisites: Mapping[str, Tuple[str, ...]]
    parallelizable: Tuple[Tuple[str, ...], ...]
    next_actions: Tuple[str, ...]
    required_approvals: Tuple[str, ...]
    stop_effects: Tuple[str, ...] = field(default_factory=tuple)
