"""Append-only JSONL workflow audit log."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from .types import AuditEvent


class AppendOnlyAuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: AuditEvent) -> None:
        encoded = json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def read_all(self) -> tuple[dict, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        )
