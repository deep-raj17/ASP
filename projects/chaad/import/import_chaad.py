"""Idempotent CHAAD evidence and registry import through ROS service interfaces."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ros.cli.services import RosServices
from ros.core.evidence_engine.engine import EvidenceEngine
from ros.core.evidence_engine.hashing import HASH_VERSION, hash_file, hash_structured
from ros.core.evidence_engine.store import AppendOnlyEvidenceStore
from ros.core.evidence_engine.types import (
    ChecksumRecord,
    EvidenceRecord,
    EvidenceSource,
    EvidenceState,
    EvidenceType,
    ProvenanceRecord,
)
from ros.core.registry import RegistryRecord
from ros.core.workflow_engine.loader import load_workflow
from ros.core.workflow_engine.types import TransitionRequest

CORRELATION_ID = "ros-project-01-chaad-20260724"
IMPORTED_AT = "2026-07-24T09:52:54Z"


def utc_from_file(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def registry_record(
    registry: str,
    record_id: str,
    entity_id: str,
    status: str,
    payload: dict,
    *,
    parents: tuple[str, ...] = (),
    supersedes: str | None = None,
) -> RegistryRecord:
    return RegistryRecord(
        registry=registry,
        record_id=record_id,
        entity_id=entity_id,
        version="1.0.0",
        schema_version="1.0.0",
        event_type="created" if not supersedes else "superseded",
        author="ros-project-01",
        timestamp=IMPORTED_AT,
        correlation_id=CORRELATION_ID,
        causation_id="chaad-migration",
        status=status,
        payload=payload,
        parent_references=parents,
        supersedes=supersedes,
        idempotency_key=f"chaad-import:{record_id}",
    )


def run(source: Path, workspace: Path, *, dry_run: bool) -> dict:
    if dry_run:
        # Validate the complete write/evaluation sequence in a disposable ROS
        # workspace. The selected workspace remains byte-for-byte untouched.
        with tempfile.TemporaryDirectory(prefix="ros-chaad-dry-run-") as temporary:
            result = run(source, Path(temporary), dry_run=False)
        result["dry_run"] = True
        result["validation_mode"] = "ephemeral-workspace"
        return result
    service = RosServices(workspace)
    evidence_store = AppendOnlyEvidenceStore(workspace / ".ros" / "evidence.jsonl")
    evidence_engine = EvidenceEngine(evidence_store)
    mapping = yaml.safe_load(
        (source / "projects/chaad/import/evidence_map.yaml").read_text(encoding="utf-8")
    )
    evidence_results = []
    requirement_map: dict[str, list[str]] = {}
    existing_verifications = evidence_store.verifications()
    for item in mapping["records"]:
        path = source / item["path"]
        metadata = {
            "relative_path": item["path"],
            "requirements": item.get("requirements", []),
            "classification": item.get("classification", "ACCEPTED"),
            "limitation": item.get("limitation"),
        }
        record = EvidenceRecord(
            evidence_id=item["id"],
            evidence_type=EvidenceType(item["type"]),
            state=EvidenceState.COLLECTED,
            created_at=utc_from_file(path),
            collected_at=IMPORTED_AT,
            project_reference="chaad",
            provenance=ProvenanceRecord(
                producer="legacy-chaad-repository",
                producer_type="project",
                source=EvidenceSource(str(path.resolve()), item["path"]),
                environment_identity="chaad-local-import",
                tool_version="ros-project-01/1.0.0",
                repository_revision="9c722c0599e32df056263573f00bce2be658a013",
                workflow_reference="PMPS@1.0.0",
            ),
            content_checksum=ChecksumRecord("sha256", HASH_VERSION, hash_file(path)),
            metadata_checksum=ChecksumRecord("sha256", HASH_VERSION, hash_structured(metadata)),
            metadata=metadata,
        )
        evidence_engine.register(record, dry_run=dry_run)
        verifications = []
        for verifier in item["verifiers"]:
            previous = next(
                (
                    value for value in existing_verifications.values()
                    if value["evidence_id"] == item["id"] and value["verifier_id"] == verifier
                ),
                None,
            )
            if previous and not dry_run:
                verifications.append({"verifier": verifier, "passed": previous["passed"], "idempotent": True})
                continue
            from ros.core.evidence_engine.types import VerificationRequest
            result = evidence_engine.verify(
                VerificationRequest(
                    item["id"], verifier, CORRELATION_ID,
                    f"chaad:{item['id']}:{verifier}", dry_run,
                )
            )
            verifications.append(
                {"verifier": verifier, "passed": result.passed, "reason": result.reason}
            )
        if all(value["passed"] for value in verifications):
            for requirement in item.get("requirements", []):
                requirement_map.setdefault(requirement, []).append(item["id"])
        evidence_results.append(
            {"evidence_id": item["id"], "checksum": record.content_checksum.value, "verifications": verifications}
        )

    entities = (
        registry_record(
            "policies", "chaad-policy-v1", "chaad-import-integrity-v1", "ACTIVE",
            {"project_id": "chaad", "path": "projects/chaad/policies.yaml"},
        ),
        registry_record(
            "datasets", "chaad-dataset-local-v1", "mimii-local-snapshot", "INCOMPLETE",
            {
                "project_id": "chaad", "manifest": "chaad-dataset-manifest",
                "version": "UNVERIFIED_LOCAL_IDENTITY", "file_count": 53046,
                "license_candidate": "CC-BY-SA-4.0", "doi_candidate": "10.5281/zenodo.3384388",
            },
        ),
        registry_record(
            "artifacts", "chaad-split-v1", "chaad-machine-independent-split", "VERIFIED",
            {
                "project_id": "chaad", "dataset_id": "mimii-local-snapshot",
                "train": ["id_04"], "validation": ["id_00", "id_02"], "test": ["id_06"],
                "evidence_id": "chaad-dataset-manifest",
            },
            parents=("chaad-dataset-local-v1",),
        ),
        registry_record(
            "experiments", "chaad-exp-001-v1", "EXP-CHAAD-001", "INCOMPLETE",
            {
                "project_id": "chaad", "workflow_id": "PMPS",
                "result_scope": "provisional-validation-only",
                "underfitting": True, "test_evaluation": "NOT_RUN",
            },
            parents=("chaad-split-v1",),
        ),
        registry_record(
            "models", "chaad-model-best-v1", "chaad-best-model", "PROVISIONAL",
            {
                "project_id": "chaad", "experiment_id": "EXP-CHAAD-001",
                "checkpoint_sha256": "7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9",
                "selected_epoch": 6,
            },
            parents=("chaad-exp-001-v1",),
        ),
        registry_record(
            "artifacts", "chaad-validation-invalid-v1", "validation-export-original", "INVALID",
            {
                "project_id": "chaad", "experiment_id": "EXP-CHAAD-001",
                "evidence_id": "chaad-invalid-validation-export",
                "reason": "30 duplicated sample IDs; 60 affected rows",
            },
            parents=("chaad-exp-001-v1",),
        ),
        registry_record(
            "artifacts", "chaad-validation-corrected-v1", "validation-export-corrected", "PROVISIONAL",
            {
                "project_id": "chaad", "experiment_id": "EXP-CHAAD-001",
                "evidence_id": "chaad-corrected-validation-export",
                "roc_auc": 0.6002609445, "scope": "validation-only",
            },
            parents=("chaad-validation-invalid-v1",),
            supersedes="chaad-validation-invalid-v1",
        ),
        registry_record(
            "publications", "chaad-publication-baseline-v1", "chaad-publication", "INCOMPLETE",
            {
                "project_id": "chaad", "publication_ready": False,
                "evidence_id": "chaad-publication-baseline",
            },
            parents=("chaad-exp-001-v1",),
        ),
        registry_record(
            "workflows", "chaad-pmps-instance-v1", "chaad-pmps-1", "BLOCKED",
            {
                "project_id": "chaad", "workflow_id": "PMPS", "workflow_version": "1.0.0",
                "gate_id": "PMPS-01", "derived_state": "BLOCKED",
            },
            parents=("chaad-dataset-local-v1", "chaad-exp-001-v1"),
        ),
    )
    entity_results = [service.registry.append(record, dry_run=dry_run) for record in entities]

    definition = load_workflow(source / "ros/specs/workflows/pmps-1.0.0.yaml")
    service.workflow.register_definition(definition)
    instances = service.workflow.store.load_all()
    if "chaad-pmps-1" not in instances:
        instance = service.workflow.create_instance(definition, "chaad-pmps-1", dry_run=dry_run)
        if not dry_run:
            service.workflow.transition(
                TransitionRequest(
                    "chaad-pmps-1", "start", "ros-project-01", "service",
                    CORRELATION_ID, "chaad-pmps-start", instance.revision,
                )
            )
    if dry_run:
        workflow_state = "BLOCKED"
        gate_state = "BLOCKED"
    else:
        instance = service.workflow.store.get("chaad-pmps-1")
        raw = yaml.safe_load((source / "ros/specs/workflows/pmps-1.0.0.yaml").read_text(encoding="utf-8"))
        all_requirements = {
            item["id"]: tuple(requirement_map.get(item["id"], ()))
            for item in raw["gates"][0]["requirements"]
        }
        outcomes = {}
        verification_ids = set()
        for requirement, evidence_ids in all_requirements.items():
            matches = [
                value for value in evidence_store.verifications().values()
                if value["evidence_id"] in evidence_ids
            ]
            verification_ids.update(value["execution_id"] for value in matches)
            outcomes[requirement] = (
                "INCOMPLETE" if not evidence_ids else
                "BLOCKED" if not matches else
                "SATISFIED" if all(value["passed"] for value in matches) else
                "UNSATISFIED"
            )
        expected_checksum = hash_structured(
            {
                "gate": "PMPS-01",
                "outcomes": outcomes,
                "evidence": sorted({item for values in all_requirements.values() for item in values}),
                "verifications": sorted(verification_ids),
            }
        )
        prior_evaluations = [
            event for event in evidence_store.events()
            if event["type"] == "GateEvidenceEvaluated"
        ]
        already_evaluated = any(
            event["result"]["evaluation_checksum"] == expected_checksum
            for event in prior_evaluations
        )
        if not already_evaluated:
            evaluation, workflow_input = evidence_engine.evaluate_gate(
                "PMPS-01", all_requirements, CORRELATION_ID
            )
            service.workflow.transition(
                TransitionRequest(
                    "chaad-pmps-1", "evaluate_gate", "ros.evidence-engine", "service",
                    CORRELATION_ID, "chaad-pmps01-evaluation", instance.revision,
                    evaluation=workflow_input,
                )
            )
        instance = service.workflow.store.get("chaad-pmps-1")
        workflow_state = instance.state.value
        gate_state = instance.gates["PMPS-01"].state.value
    return {
        "dry_run": dry_run,
        "correlation_id": CORRELATION_ID,
        "evidence": evidence_results,
        "registry": entity_results,
        "gate_state": gate_state,
        "workflow_state": workflow_state,
        "registry_integrity": service.verify_registry(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.source).resolve(), Path(args.workspace).resolve(), dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
