"""Gate-evaluation contract validation."""

from __future__ import annotations

from .errors import ErrorCode, WorkflowError
from .types import EvaluationResult, GateDefinition, GateEvaluationInput


def validate_gate_evaluation(
    definition: GateDefinition, evaluation: GateEvaluationInput
) -> None:
    if evaluation.gate_id != definition.id:
        raise WorkflowError(
            ErrorCode.INVALID_TRANSITION,
            f"Evaluation gate {evaluation.gate_id} does not match {definition.id}",
        )
    required = (
        evaluation.evaluator_identity,
        evaluation.evaluator_version,
        evaluation.timestamp,
        evaluation.verification_checksum,
        evaluation.correlation_id,
    )
    if not all(required):
        raise WorkflowError(
            ErrorCode.INVALID_TRANSITION,
            "Gate evaluation is missing required verifier metadata",
        )
    if (
        not evaluation.evidence_references
        and not definition.administrative
        and evaluation.result is not EvaluationResult.WAIVED
    ):
        raise WorkflowError(
            ErrorCode.EVIDENCE_REFERENCE_REQUIRED,
            f"Gate {definition.id} requires evidence references",
        )
    if evaluation.result is EvaluationResult.WAIVED:
        if not definition.allow_waiver:
            raise WorkflowError(
                ErrorCode.POLICY_VIOLATION,
                f"Gate {definition.id} does not permit waiver",
            )
        if not definition.waiver_policy or definition.waiver_policy not in evaluation.policy_references:
            raise WorkflowError(
                ErrorCode.POLICY_VIOLATION,
                f"Waiver policy {definition.waiver_policy} is required",
            )
        if not evaluation.approval_references:
            raise WorkflowError(
                ErrorCode.APPROVAL_REQUIRED,
                "Waiver requires a policy-authorized approval reference",
            )
