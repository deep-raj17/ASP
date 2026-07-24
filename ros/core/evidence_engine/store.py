from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any


class AppendOnlyEvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: dict[str, Any], dry_run: bool = False) -> None:
        if dry_run:
            return
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def events(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        )

    def records(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for event in self.events():
            if event["type"] == "EvidenceRegistered":
                result[event["record"]["evidence_id"]] = event["record"]
        return result

    def verifications(self) -> dict[str, dict[str, Any]]:
        result = {}
        for event in self.events():
            if event["type"] in {"EvidenceVerified", "EvidenceVerificationFailed"}:
                result[event["result"]["execution_id"]] = event["result"]
        return result
