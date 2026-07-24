import json

import pytest

from ros.core.evidence_engine.hashing import hash_structured
from ros.core.registry import RegistryRecord, SQLiteRegistry


def rec(record_id, entity="entity", version="1.0.0", **kwargs):
    return RegistryRecord(
        registry=kwargs.get("registry", "projects"),
        record_id=record_id,
        entity_id=entity,
        version=version,
        schema_version="1.0.0",
        event_type=kwargs.get("event_type", "created"),
        author="tester",
        timestamp="2026-07-24T00:00:00Z",
        correlation_id="corr",
        causation_id="cause",
        status=kwargs.get("status", "ACTIVE"),
        payload=kwargs.get("payload", {"value": record_id}),
        parent_references=kwargs.get("parents", ()),
        supersedes=kwargs.get("supersedes"),
        idempotency_key=kwargs.get("key", record_id),
    )


def test_append_history_current_and_no_mutation(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    first = registry.append(rec("r1"))
    assert first["sequence"] == 1
    assert registry.append(rec("r1"))["idempotent"]
    second = registry.supersede("r1", rec("r2", version="2.0.0", key="r2"))
    assert second["sequence"] == 2
    assert len(registry.history("projects", "entity")) == 2
    assert registry.current_view("projects")[0]["record_id"] == "r2"
    with registry._connect() as db:
        with pytest.raises(Exception, match="UNAUTHORIZED_MUTATION"):
            db.execute("UPDATE records SET status='X' WHERE record_id='r1'")
        with pytest.raises(Exception, match="UNAUTHORIZED_MUTATION"):
            db.execute("DELETE FROM records WHERE record_id='r1'")


def test_integrity_export_import_and_restart(tmp_path):
    source = SQLiteRegistry(tmp_path / "source.db")
    source.append(rec("failed", registry="experiments", status="FAILED"))
    source.append(rec("incomplete", entity="e2", registry="experiments", status="INCOMPLETE"))
    report = source.verify_integrity()
    assert report.valid and report.record_count == 2
    bundle = tmp_path / "bundle.json"
    source.export_bundle(bundle)
    target = SQLiteRegistry(tmp_path / "target.db")
    preview = target.import_bundle(bundle, dry_run=True)
    assert len(preview) == 2 and target.history() == ()
    target.import_bundle(bundle, dry_run=False)
    assert target.verify_integrity().head_checksum == report.head_checksum
    restarted = SQLiteRegistry(tmp_path / "target.db")
    assert {row["status"] for row in restarted.history("experiments")} == {"FAILED", "INCOMPLETE"}


def test_references_conflicts_and_dry_run(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    preview = registry.append(rec("r1"), dry_run=True)
    assert preview["sequence"] == 1 and registry.history() == ()
    registry.append(rec("r1"), expected_sequence=0)
    with pytest.raises(ValueError, match="CONFLICTING_EVENT"):
        registry.append(rec("r1-reused", payload={"different": True}, key="r1"))
    with pytest.raises(ValueError, match="CONCURRENCY_CONFLICT"):
        registry.append(rec("r2", version="2.0.0", key="r2"), expected_sequence=0)
    with pytest.raises(ValueError, match="BROKEN_REFERENCE"):
        registry.append(rec("r3", entity="x", parents=("missing",), key="r3"))


def test_tombstone_hides_current_but_preserves_history(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    registry.append(rec("r1"))
    registry.tombstone("r1", rec("r2", version="2.0.0", key="r2"))
    assert registry.current_view("projects") == ()
    assert len(registry.history("projects")) == 2


def test_lifecycle_latest_and_reference_queries(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    registry.append(rec("project", entity="chaad", payload={"project_id": "chaad"}))
    registry.append(
        rec(
            "experiment",
            entity="exp-1",
            registry="experiments",
            payload={"project_id": "chaad", "workflow_id": "pmps", "experiment_id": "exp-1"},
            parents=("project",),
        )
    )
    registry.deprecate(
        "experiment",
        rec("experiment-deprecated", entity="exp-1", version="2.0.0", registry="experiments"),
    )
    assert registry.latest_valid("projects", "chaad")["record_id"] == "project"
    assert registry.latest_valid("experiments", "exp-1")["record_id"] == "experiment"
    assert registry.current_view("experiments") == ()
    assert registry.query_by_reference("project")[0]["record_id"] == "experiment"
    assert len(registry.query_by_project("chaad")) == 2
    assert registry.query_by_workflow("pmps")[0]["record_id"] == "experiment"
    assert registry.query_by_experiment("exp-1")[0]["record_id"] == "experiment"
    rebuilt = registry.rebuild_view("experiments")
    assert rebuilt["source_position"] == 3


def test_integrity_detects_metadata_tampering(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.db")
    registry.append(rec("r1"))
    with registry._connect() as db:
        db.execute("DROP TRIGGER records_no_update")
        db.execute("UPDATE records SET author='attacker' WHERE record_id='r1'")
    report = registry.verify_integrity()
    assert not report.valid
    assert any(issue.code == "CHECKSUM_MISMATCH" and issue.message == "Metadata changed"
               for issue in report.issues)


def test_import_rejects_tamper_before_mutation_and_imports_lineage(tmp_path):
    source = SQLiteRegistry(tmp_path / "source.db")
    source.append(rec("parent"))
    source.append(rec("child", entity="child", parents=("parent",)))
    bundle = tmp_path / "bundle.json"
    source.export_bundle(bundle)

    target = SQLiteRegistry(tmp_path / "target.db")
    target.import_bundle(bundle, dry_run=False)
    assert len(target.history()) == 2
    assert target.verify_integrity().valid

    raw = json.loads(bundle.read_text(encoding="utf-8"))
    raw["records"][1]["author"] = "attacker"
    body = {key: value for key, value in raw.items() if key != "manifest_checksum"}
    raw["manifest_checksum"] = hash_structured(body)
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    clean = SQLiteRegistry(tmp_path / "clean.db")
    with pytest.raises(ValueError, match="IMPORT_VALIDATION_FAILED"):
        clean.import_bundle(tampered, dry_run=False)
    assert clean.history() == ()


def test_integration_cross_registry_history_survives_restart(tmp_path):
    path = tmp_path / "registry.db"
    registry = SQLiteRegistry(path)
    registry.append(rec("project", entity="chaad", payload={"project_id": "chaad"}))
    registry.append(rec(
        "workflow", entity="pmps-run", registry="workflows", parents=("project",),
        payload={"project_id": "chaad", "workflow_id": "pmps"},
    ))
    registry.append(rec(
        "evidence", entity="manifest", registry="artifacts", parents=("workflow",),
        payload={"project_id": "chaad", "workflow_id": "pmps"},
    ))
    registry.append(rec(
        "failed-experiment", entity="exp-1", registry="experiments", status="FAILED",
        parents=("evidence",), payload={"project_id": "chaad", "experiment_id": "exp-1"},
    ))
    registry.append(rec(
        "model", entity="model-1", registry="models", parents=("failed-experiment",),
        payload={"project_id": "chaad", "experiment_id": "exp-1"},
    ))
    registry.append(rec(
        "publication", entity="paper-1", registry="publications", parents=("evidence",),
        payload={"project_id": "chaad"},
    ))
    restarted = SQLiteRegistry(path)
    assert restarted.verify_integrity().valid
    assert restarted.get("failed-experiment")["status"] == "FAILED"
    assert restarted.get("model")["parent_references"] == ("failed-experiment",)
