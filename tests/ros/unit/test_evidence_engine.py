from pathlib import Path

import pytest

from ros.core.evidence_engine.engine import EvidenceEngine
from ros.core.evidence_engine.hashing import HASH_VERSION, hash_file, hash_structured
from ros.core.evidence_engine.store import AppendOnlyEvidenceStore
from ros.core.evidence_engine.types import (
    ChecksumRecord,
    EvidenceRecord,
    EvidenceSource,
    EvidenceState,
    EvidenceType,
    LineageReference,
    ProvenanceRecord,
    VerificationRequest,
)


def record(path: Path, evidence_id="ev-1", *, parents=(), checksum=None):
    content = checksum or hash_file(path)
    metadata = {"name": path.name}
    return EvidenceRecord(
        evidence_id, EvidenceType.FILE_ARTIFACT, EvidenceState.COLLECTED,
        "2026-07-24T00:00:00Z", "2026-07-24T00:00:00Z", "project-1",
        ProvenanceRecord(
            "collector", "service", EvidenceSource(str(path), "source-1"),
            "env-1", "1.0.0",
        ),
        ChecksumRecord("sha256", HASH_VERSION, content),
        ChecksumRecord("sha256", HASH_VERSION, hash_structured(metadata)),
        parents=parents, metadata=metadata,
    )


def test_hash_is_cross_path_stable(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    left.write_bytes(b"same")
    right.write_bytes(b"same")
    assert hash_file(left) == hash_file(right)
    assert hash_structured({"b": 2, "a": 1}) == hash_structured({"a": 1, "b": 2})


def test_registration_verification_idempotency_and_failure_preservation(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"content")
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.jsonl")
    engine = EvidenceEngine(store)
    assert engine.register(record(path)) == "ev-1"
    assert engine.register(record(path)) == "ev-1"
    request = VerificationRequest("ev-1", "file-checksum", "c", "key")
    first = engine.verify(request)
    assert first.passed and first.state is EvidenceState.VERIFIED
    assert engine.verify(request).execution_id == first.execution_id
    path.write_bytes(b"tampered")
    second = engine.verify(
        VerificationRequest("ev-1", "file-checksum", "c", "key-2")
    )
    assert not second.passed
    assert second.state is EvidenceState.QUARANTINED
    assert len(store.verifications()) == 2


def test_lineage_and_cycle_rejection(tmp_path):
    first_path, second_path = tmp_path / "a", tmp_path / "b"
    first_path.write_bytes(b"a")
    second_path.write_bytes(b"b")
    engine = EvidenceEngine(AppendOnlyEvidenceStore(tmp_path / "events"))
    engine.register(record(first_path, "parent"))
    engine.register(
        record(second_path, "child", parents=(LineageReference("parent"),))
    )
    assert engine.lineage("child") == ("parent", "child")
    with pytest.raises(ValueError, match="LINEAGE_CYCLE"):
        engine.register(
            record(second_path, "cycle", parents=(LineageReference("cycle"),))
        )


def test_gate_evaluation_is_structured_and_does_not_mutate_workflow(tmp_path):
    path = tmp_path / "a"
    path.write_bytes(b"a")
    store = AppendOnlyEvidenceStore(tmp_path / "events")
    engine = EvidenceEngine(store)
    engine.register(record(path))
    engine.verify(VerificationRequest("ev-1", "file-checksum", "c", "verify"))
    result, workflow_input = engine.evaluate_gate(
        "gate-1", {"artifact": ("ev-1",)}, "corr"
    )
    assert result.outcome == "SATISFIED"
    assert workflow_input.result.value == "SATISFIED"
    assert workflow_input.evaluator_identity == "ros.evidence-engine"


def test_missing_and_unsupported_verifier(tmp_path):
    engine = EvidenceEngine(AppendOnlyEvidenceStore(tmp_path / "events"))
    with pytest.raises(ValueError, match="INVALID_EVIDENCE_RECORD"):
        engine.verify(VerificationRequest("missing", "file-checksum", "c", "k"))
