from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone
from typing import Iterable

from ros.core.workflow_engine.types import EvaluationResult, GateEvaluationInput

from .hashing import HASH_VERSION, hash_structured
from .store import AppendOnlyEvidenceStore
from .types import (
    ChecksumRecord,
    EvidenceRecord,
    EvidenceState,
    GateEvidenceEvaluation,
    VerificationRequest,
    VerificationResult,
)
from .verifiers import BUILTINS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceEngine:
    def __init__(self, store: AppendOnlyEvidenceStore):
        self.store = store
        self._verification_keys: dict[str, VerificationResult] = {}

    def register(self, record: EvidenceRecord, *, dry_run: bool = False) -> str:
        current = self.store.records()
        if record.evidence_id in current:
            if current[record.evidence_id]["content_checksum"]["value"] == record.content_checksum.value:
                return record.evidence_id
            raise ValueError("DUPLICATE_EVIDENCE")
        self._assert_acyclic(record)
        for parent in record.parents:
            if parent.evidence_id not in current:
                raise ValueError("MISSING_PARENT_EVIDENCE")
        self.store.append(
            {"type": "EvidenceRegistered", "timestamp": now(), "record": dataclasses.asdict(record)},
            dry_run,
        )
        return record.evidence_id

    def verify(self, request: VerificationRequest) -> VerificationResult:
        if request.idempotency_key in self._verification_keys:
            return self._verification_keys[request.idempotency_key]
        raw = self.store.records().get(request.evidence_id)
        if not raw:
            raise ValueError("INVALID_EVIDENCE_RECORD")
        record = self._decode_record(raw)
        if request.verifier_id not in BUILTINS:
            raise ValueError("VERIFIER_NOT_FOUND")
        supported, verifier = BUILTINS[request.verifier_id]
        if record.evidence_type not in supported:
            raise ValueError("UNSUPPORTED_EVIDENCE_TYPE")
        passed, reason, details = verifier(record)
        state = EvidenceState.VERIFIED if passed else (
            EvidenceState.QUARANTINED if "checksum mismatch" in reason else EvidenceState.FAILED_VERIFICATION
        )
        body = {
            "evidence": record.evidence_id,
            "verifier": request.verifier_id,
            "passed": passed,
            "reason": reason,
            "details": details,
        }
        result = VerificationResult(
            str(uuid.uuid4()), record.evidence_id, request.verifier_id, "1.0.0",
            state, passed, reason, details,
            ChecksumRecord("sha256", HASH_VERSION, hash_structured(body)), now(),
        )
        if not request.dry_run:
            self.store.append(
                {
                    "type": "EvidenceVerified" if passed else "EvidenceVerificationFailed",
                    "timestamp": result.timestamp,
                    "result": dataclasses.asdict(result),
                }
            )
            self._verification_keys[request.idempotency_key] = result
        return result

    def lineage(self, evidence_id: str) -> tuple[str, ...]:
        records = self.store.records()
        if evidence_id not in records:
            raise ValueError("INVALID_EVIDENCE_RECORD")
        ordered: list[str] = []
        seen: set[str] = set()
        def walk(current: str) -> None:
            if current in seen:
                return
            seen.add(current)
            for parent in records[current].get("parents", []):
                parent_id = parent["evidence_id"]
                if parent_id not in records:
                    raise ValueError("MISSING_PARENT_EVIDENCE")
                walk(parent_id)
            ordered.append(current)
        walk(evidence_id)
        return tuple(ordered)

    def evaluate_gate(
        self,
        gate_id: str,
        requirements: dict[str, tuple[str, ...]],
        correlation_id: str,
    ) -> tuple[GateEvidenceEvaluation, GateEvaluationInput]:
        verifications = self.store.verifications()
        outcomes = {}
        evidence_refs, verification_refs = set(), set()
        for requirement, evidence_ids in requirements.items():
            if not evidence_ids:
                outcomes[requirement] = "INCOMPLETE"
                continue
            matching = [
                item for item in verifications.values()
                if item["evidence_id"] in evidence_ids
            ]
            evidence_refs.update(evidence_ids)
            verification_refs.update(item["execution_id"] for item in matching)
            if not matching:
                outcomes[requirement] = "BLOCKED"
            elif all(item["passed"] for item in matching):
                outcomes[requirement] = "SATISFIED"
            else:
                outcomes[requirement] = "UNSATISFIED"
        overall = (
            "UNSATISFIED" if "UNSATISFIED" in outcomes.values()
            else "BLOCKED" if {"BLOCKED", "INCOMPLETE"} & set(outcomes.values())
            else "SATISFIED"
        )
        payload = {
            "gate": gate_id, "outcomes": outcomes,
            "evidence": sorted(evidence_refs), "verifications": sorted(verification_refs),
        }
        checksum = hash_structured(payload)
        result = GateEvidenceEvaluation(
            gate_id, overall, tuple(sorted(evidence_refs)),
            tuple(sorted(verification_refs)), outcomes, (), checksum,
            "1.0.0", now(), correlation_id,
        )
        workflow_input = GateEvaluationInput(
            gate_id, EvaluationResult(overall), result.evidence_references,
            "ros.evidence-engine", result.evaluator_version, (), result.timestamp,
            checksum, correlation_id,
        )
        self.store.append({"type": "GateEvidenceEvaluated", "timestamp": now(), "result": dataclasses.asdict(result)})
        return result, workflow_input

    def _assert_acyclic(self, candidate: EvidenceRecord) -> None:
        if any(parent.evidence_id == candidate.evidence_id for parent in candidate.parents):
            raise ValueError("LINEAGE_CYCLE")

    @staticmethod
    def _decode_record(raw: dict) -> EvidenceRecord:
        from .types import ChecksumRecord, EvidenceSource, EvidenceType, LineageReference, ProvenanceRecord
        provenance = raw["provenance"]
        return EvidenceRecord(
            raw["evidence_id"], EvidenceType(raw["evidence_type"]), EvidenceState(raw["state"]),
            raw["created_at"], raw["collected_at"], raw["project_reference"],
            ProvenanceRecord(
                provenance["producer"], provenance["producer_type"],
                EvidenceSource(**provenance["source"]), provenance["environment_identity"],
                provenance["tool_version"], provenance.get("repository_revision"),
                provenance.get("workflow_reference"), provenance.get("experiment_reference"),
            ),
            ChecksumRecord(**raw["content_checksum"]),
            ChecksumRecord(**raw["metadata_checksum"]),
            tuple(LineageReference(**item) for item in raw.get("parents", [])),
            raw.get("access_classification", "internal"),
            raw.get("retention_policy", "indefinite"),
            raw.get("metadata") or {},
            raw.get("supersedes"),
        )
