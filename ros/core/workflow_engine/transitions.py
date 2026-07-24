"""Legal workflow transition table."""

from __future__ import annotations

from .types import WorkflowState

ALLOWED_WORKFLOW_TRANSITIONS = {
    WorkflowState.NOT_STARTED: {WorkflowState.READY, WorkflowState.CANCELLED},
    WorkflowState.READY: {
        WorkflowState.RUNNING,
        WorkflowState.BLOCKED,
        WorkflowState.CANCELLED,
    },
    WorkflowState.RUNNING: {
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
        WorkflowState.COMPLETED,
        WorkflowState.CANCELLED,
    },
    WorkflowState.BLOCKED: {
        WorkflowState.READY,
        WorkflowState.RUNNING,
        WorkflowState.FAILED,
        WorkflowState.CANCELLED,
    },
    WorkflowState.FAILED: {WorkflowState.READY},
    WorkflowState.COMPLETED: set(),
    WorkflowState.CANCELLED: set(),
}


def is_workflow_transition_allowed(
    previous: WorkflowState, requested: WorkflowState
) -> bool:
    return requested in ALLOWED_WORKFLOW_TRANSITIONS[previous]
