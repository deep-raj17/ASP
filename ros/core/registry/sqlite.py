from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ros.core.evidence_engine.hashing import canonical_bytes, hash_structured

from .types import IntegrityIssue, IntegrityReport, REGISTRY_NAMES, RegistryRecord

GENESIS = "0" * 64


class SQLiteRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS records (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                  registry TEXT NOT NULL,
                  record_id TEXT NOT NULL UNIQUE,
                  entity_id TEXT NOT NULL,
                  version TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  author TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  correlation_id TEXT NOT NULL,
                  causation_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  parent_references TEXT NOT NULL,
                  supersedes TEXT,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  content_checksum TEXT NOT NULL,
                  metadata_checksum TEXT NOT NULL,
                  previous_checksum TEXT NOT NULL,
                  record_checksum TEXT NOT NULL,
                  UNIQUE(registry, entity_id, version)
                );
                CREATE TRIGGER IF NOT EXISTS records_no_update
                BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'UNAUTHORIZED_MUTATION'); END;
                CREATE TRIGGER IF NOT EXISTS records_no_delete
                BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'UNAUTHORIZED_MUTATION'); END;
                """
            )

    def append(
        self,
        record: RegistryRecord,
        *,
        expected_sequence: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._validate(record)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT sequence, record_id, content_checksum FROM records WHERE idempotency_key=?",
                    (record.idempotency_key,),
                ).fetchone()
                if row:
                    requested_checksum = hashlib.sha256(
                        json.dumps(record.payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    if requested_checksum != row["content_checksum"]:
                        raise ValueError("CONFLICTING_EVENT")
                    db.execute("ROLLBACK")
                    return {"sequence": row["sequence"], "record_id": row["record_id"], "idempotent": True}
                head = db.execute(
                    "SELECT sequence, record_checksum FROM records ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                actual_sequence = int(head["sequence"]) if head else 0
                previous = str(head["record_checksum"]) if head else GENESIS
                if expected_sequence is not None and expected_sequence != actual_sequence:
                    raise ValueError("CONCURRENCY_CONFLICT")
                self._validate_references(db, record)
                payload_json = json.dumps(record.payload, sort_keys=True, separators=(",", ":"))
                parents_json = json.dumps(sorted(record.parent_references), separators=(",", ":"))
                content_checksum = hashlib.sha256(payload_json.encode()).hexdigest()
                metadata = {
                    key: value for key, value in asdict(record).items()
                    if key not in {"payload"}
                }
                metadata_checksum = hash_structured(metadata)
                record_checksum = hash_structured(
                    {
                        "previous": previous,
                        "content": content_checksum,
                        "metadata": metadata_checksum,
                        "sequence": actual_sequence + 1,
                    }
                )
                result = {
                    "sequence": actual_sequence + 1,
                    "record_id": record.record_id,
                    "idempotent": False,
                    "content_checksum": content_checksum,
                    "metadata_checksum": metadata_checksum,
                    "previous_checksum": previous,
                    "record_checksum": record_checksum,
                }
                if dry_run:
                    db.execute("ROLLBACK")
                    return result
                db.execute(
                    """INSERT INTO records (
                    registry,record_id,entity_id,version,schema_version,event_type,author,
                    timestamp,correlation_id,causation_id,status,payload,parent_references,
                    supersedes,idempotency_key,content_checksum,metadata_checksum,
                    previous_checksum,record_checksum) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record.registry, record.record_id, record.entity_id, record.version,
                        record.schema_version, record.event_type, record.author, record.timestamp,
                        record.correlation_id, record.causation_id, record.status, payload_json,
                        parents_json, record.supersedes, record.idempotency_key, content_checksum,
                        metadata_checksum, previous, record_checksum,
                    ),
                )
                db.execute("COMMIT")
                return result
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def get(self, record_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
        if not row:
            raise KeyError("RECORD_NOT_FOUND")
        return self._row(row)

    def get_version(self, registry: str, entity_id: str, version: str):
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM records WHERE registry=? AND entity_id=? AND version=?",
                (registry, entity_id, version),
            ).fetchone()
        if not row:
            raise KeyError("VERSION_NOT_FOUND")
        return self._row(row)

    def history(self, registry: Optional[str] = None, entity_id: Optional[str] = None):
        query, params = "SELECT * FROM records WHERE 1=1", []
        if registry:
            query += " AND registry=?"; params.append(registry)
        if entity_id:
            query += " AND entity_id=?"; params.append(entity_id)
        query += " ORDER BY sequence"
        with self._connect() as db:
            return tuple(self._row(row) for row in db.execute(query, params))

    def current_view(self, registry: str):
        records = self.history(registry)
        current: dict[str, dict[str, Any]] = {}
        for item in records:
            if item["event_type"] in {"deprecated", "tombstone", "revoked"}:
                current.pop(item["entity_id"], None)
            else:
                current[item["entity_id"]] = item
        return tuple(current[key] for key in sorted(current))

    def latest_valid(self, registry: str, entity_id: str):
        candidates = [
            item for item in self.history(registry, entity_id)
            if item["event_type"] not in {"tombstone", "revoked"}
            and item["status"] not in {"DEPRECATED", "REVOKED"}
        ]
        if not candidates:
            raise KeyError("VERSION_NOT_FOUND")
        return candidates[-1]

    def supersede(self, previous_id: str, new_record: RegistryRecord, *, dry_run=False):
        self.get(previous_id)
        return self.append(replace(new_record, supersedes=previous_id), dry_run=dry_run)

    def deprecate(self, previous_id: str, record: RegistryRecord, *, dry_run=False):
        previous = self.get(previous_id)
        if record.entity_id != previous["entity_id"] or record.registry != previous["registry"]:
            raise ValueError("CONFLICTING_EVENT")
        return self.append(
            replace(record, event_type="deprecated", status="DEPRECATED", supersedes=previous_id),
            dry_run=dry_run,
        )

    def revoke(self, previous_id: str, record: RegistryRecord, *, dry_run=False):
        previous = self.get(previous_id)
        if record.entity_id != previous["entity_id"] or record.registry != previous["registry"]:
            raise ValueError("CONFLICTING_EVENT")
        return self.append(
            replace(record, event_type="revoked", status="REVOKED", supersedes=previous_id),
            dry_run=dry_run,
        )

    def tombstone(self, previous_id: str, record: RegistryRecord, *, dry_run=False):
        previous = self.get(previous_id)
        if record.entity_id != previous["entity_id"] or record.registry != previous["registry"]:
            raise ValueError("INVALID_TOMBSTONE")
        return self.append(
            replace(record, event_type="tombstone", supersedes=previous_id),
            dry_run=dry_run,
        )

    def query_by_reference(self, record_id: str):
        return tuple(
            item for item in self.history()
            if record_id in item["parent_references"] or item["supersedes"] == record_id
        )

    def query_by_payload(self, field: str, value: Any):
        return tuple(item for item in self.history() if item["payload"].get(field) == value)

    def query_by_project(self, project_id: str):
        return self.query_by_payload("project_id", project_id)

    def query_by_workflow(self, workflow_id: str):
        return self.query_by_payload("workflow_id", workflow_id)

    def query_by_experiment(self, experiment_id: str):
        return self.query_by_payload("experiment_id", experiment_id)

    def verify_integrity(self) -> IntegrityReport:
        issues, previous = [], GENESIS
        records = self.history()
        known_ids: set[str] = set()
        versions: set[tuple[str, str, str]] = set()
        for expected, item in enumerate(records, 1):
            if item["sequence"] != expected:
                issues.append(IntegrityIssue(expected, "MISSING_EVENT", "Sequence gap"))
            if item["previous_checksum"] != previous:
                issues.append(IntegrityIssue(expected, "CHECKSUM_MISMATCH", "Broken hash chain"))
            content = hashlib.sha256(
                json.dumps(item["payload"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if content != item["content_checksum"]:
                issues.append(IntegrityIssue(expected, "CHECKSUM_MISMATCH", "Payload changed"))
            metadata = self._metadata_from_item(item)
            if hash_structured(metadata) != item["metadata_checksum"]:
                issues.append(IntegrityIssue(expected, "CHECKSUM_MISMATCH", "Metadata changed"))
            calculated = hash_structured(
                {"previous": item["previous_checksum"], "content": item["content_checksum"],
                 "metadata": item["metadata_checksum"], "sequence": item["sequence"]}
            )
            if calculated != item["record_checksum"]:
                issues.append(IntegrityIssue(expected, "CHECKSUM_MISMATCH", "Record digest changed"))
            identity = (item["registry"], item["entity_id"], item["version"])
            if identity in versions:
                issues.append(IntegrityIssue(expected, "DUPLICATE_VERSION", "Duplicate logical version"))
            versions.add(identity)
            for reference in item["parent_references"]:
                if reference not in known_ids:
                    issues.append(IntegrityIssue(expected, "BROKEN_REFERENCE", reference))
            if item["supersedes"] and item["supersedes"] not in known_ids:
                issues.append(IntegrityIssue(expected, "BROKEN_SUPERSESSION_CHAIN", item["supersedes"]))
            if item["event_type"] == "tombstone" and not item["supersedes"]:
                issues.append(IntegrityIssue(expected, "INVALID_TOMBSTONE", "Missing supersedes reference"))
            if item["schema_version"] != "1.0.0":
                issues.append(IntegrityIssue(expected, "UNSUPPORTED_SCHEMA_VERSION", item["schema_version"]))
            known_ids.add(item["record_id"])
            previous = item["record_checksum"]
        return IntegrityReport(not issues, len(records), previous, tuple(issues))

    def rebuild_view(self, registry: str):
        return {"registry": registry, "source_position": len(self.history()), "records": self.current_view(registry)}

    def export_bundle(self, path: str | Path, *, dry_run=False):
        records = self.history()
        body = {
            "format_version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tool_version": "1.0.0",
            "records": records,
        }
        body["manifest_checksum"] = hash_structured(body)
        if not dry_run:
            Path(path).write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return body

    def import_bundle(self, path: str | Path, *, dry_run=True):
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = body.pop("manifest_checksum", None)
        if not checksum or hash_structured(body) != checksum:
            raise ValueError("IMPORT_VALIDATION_FAILED")
        records = body.get("records")
        if body.get("format_version") != "1.0.0" or not isinstance(records, list):
            raise ValueError("IMPORT_VALIDATION_FAILED")
        self._validate_exported_records(records)
        parsed = []
        for raw in records:
            record = RegistryRecord(
                registry=raw["registry"], record_id=raw["record_id"], entity_id=raw["entity_id"],
                version=raw["version"], schema_version=raw["schema_version"],
                event_type=raw["event_type"], author=raw["author"], timestamp=raw["timestamp"],
                correlation_id=raw["correlation_id"], causation_id=raw["causation_id"],
                status=raw["status"], payload=raw["payload"],
                parent_references=tuple(raw["parent_references"]), supersedes=raw["supersedes"],
                idempotency_key=raw["idempotency_key"],
            )
            self._validate(record)
            parsed.append(record)

        # Validate every target-side conflict and reference before the first mutation.
        with self._connect() as db:
            existing_ids = {row[0] for row in db.execute("SELECT record_id FROM records")}
            existing_keys = {row[0] for row in db.execute("SELECT idempotency_key FROM records")}
            existing_versions = {
                tuple(row) for row in db.execute("SELECT registry,entity_id,version FROM records")
            }
        available_ids = set(existing_ids)
        seen_keys: set[str] = set()
        seen_versions: set[tuple[str, str, str]] = set()
        for record in parsed:
            identity = (record.registry, record.entity_id, record.version)
            if record.record_id in available_ids or record.idempotency_key in existing_keys | seen_keys:
                raise ValueError("DUPLICATE_RECORD")
            if identity in existing_versions | seen_versions:
                raise ValueError("CONFLICTING_EVENT")
            for reference in (*record.parent_references, *((record.supersedes,) if record.supersedes else ())):
                if reference not in available_ids:
                    raise ValueError("BROKEN_REFERENCE")
            available_ids.add(record.record_id)
            seen_keys.add(record.idempotency_key)
            seen_versions.add(identity)

        if dry_run:
            return tuple({"record_id": record.record_id, "validated": True} for record in parsed)

        # A clean import is committed as one transaction, so failure cannot leave a prefix.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                head = db.execute(
                    "SELECT sequence, record_checksum FROM records ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = int(head["sequence"]) if head else 0
                previous = str(head["record_checksum"]) if head else GENESIS
                imported = []
                for record in parsed:
                    sequence += 1
                    values = self._record_values(record, sequence, previous)
                    db.execute(
                        """INSERT INTO records (
                        registry,record_id,entity_id,version,schema_version,event_type,author,
                        timestamp,correlation_id,causation_id,status,payload,parent_references,
                        supersedes,idempotency_key,content_checksum,metadata_checksum,
                        previous_checksum,record_checksum) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        values["sql"],
                    )
                    previous = values["result"]["record_checksum"]
                    imported.append(values["result"])
                db.execute("COMMIT")
                return tuple(imported)
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def _validate(self, record: RegistryRecord):
        if record.registry not in REGISTRY_NAMES:
            raise ValueError("REGISTRY_NOT_FOUND")
        if not all((record.record_id, record.entity_id, record.version, record.author, record.timestamp, record.idempotency_key)):
            raise ValueError("SCHEMA_VALIDATION_FAILED")
        canonical_bytes(record.payload)

    @staticmethod
    def _metadata_from_item(item: dict[str, Any]):
        return {
            key: item[key] for key in (
                "registry", "record_id", "entity_id", "version", "schema_version",
                "event_type", "author", "timestamp", "correlation_id", "causation_id",
                "status", "parent_references", "supersedes", "idempotency_key",
            )
        }

    @classmethod
    def _validate_exported_records(cls, records: list[dict[str, Any]]):
        previous = GENESIS
        for expected, item in enumerate(records, 1):
            if item.get("sequence") != expected or item.get("previous_checksum") != previous:
                raise ValueError("IMPORT_VALIDATION_FAILED")
            payload_checksum = hashlib.sha256(
                json.dumps(item["payload"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if payload_checksum != item.get("content_checksum"):
                raise ValueError("IMPORT_VALIDATION_FAILED")
            if hash_structured(cls._metadata_from_item(item)) != item.get("metadata_checksum"):
                raise ValueError("IMPORT_VALIDATION_FAILED")
            record_checksum = hash_structured(
                {
                    "previous": previous,
                    "content": item["content_checksum"],
                    "metadata": item["metadata_checksum"],
                    "sequence": expected,
                }
            )
            if record_checksum != item.get("record_checksum"):
                raise ValueError("IMPORT_VALIDATION_FAILED")
            previous = record_checksum

    @staticmethod
    def _record_values(record: RegistryRecord, sequence: int, previous: str):
        payload_json = json.dumps(record.payload, sort_keys=True, separators=(",", ":"))
        parents_json = json.dumps(sorted(record.parent_references), separators=(",", ":"))
        content_checksum = hashlib.sha256(payload_json.encode()).hexdigest()
        metadata = {key: value for key, value in asdict(record).items() if key != "payload"}
        metadata_checksum = hash_structured(metadata)
        record_checksum = hash_structured(
            {
                "previous": previous,
                "content": content_checksum,
                "metadata": metadata_checksum,
                "sequence": sequence,
            }
        )
        result = {
            "sequence": sequence,
            "record_id": record.record_id,
            "idempotent": False,
            "content_checksum": content_checksum,
            "metadata_checksum": metadata_checksum,
            "previous_checksum": previous,
            "record_checksum": record_checksum,
        }
        sql = (
            record.registry, record.record_id, record.entity_id, record.version,
            record.schema_version, record.event_type, record.author, record.timestamp,
            record.correlation_id, record.causation_id, record.status, payload_json,
            parents_json, record.supersedes, record.idempotency_key, content_checksum,
            metadata_checksum, previous, record_checksum,
        )
        return {"result": result, "sql": sql}

    @staticmethod
    def _validate_references(db, record: RegistryRecord):
        for reference in (*record.parent_references, *((record.supersedes,) if record.supersedes else ())):
            if not db.execute("SELECT 1 FROM records WHERE record_id=?", (reference,)).fetchone():
                raise ValueError("BROKEN_REFERENCE")

    @staticmethod
    def _row(row):
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["parent_references"] = tuple(json.loads(result["parent_references"]))
        return result
