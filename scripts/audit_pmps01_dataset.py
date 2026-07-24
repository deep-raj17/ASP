"""Read-only full-corpus PMPS-01 WAV, finite-value, and SHA-256 audit.

The dataset and manifest are never modified. Progress and final reports are
new files under the caller-selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="metadata/dataset_manifest.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {output.name}")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    state = {
        "schema_version": "ros.pmps01-dataset-audit/v1",
        "started_at": utc_now(),
        "completed_at": None,
        "manifest": manifest.name,
        "manifest_sha256": manifest_hash,
        "expected_files": len(rows),
        "processed_files": 0,
        "readable_files": 0,
        "finite_files": 0,
        "sha256_match_files": 0,
        "metadata_match_files": 0,
        "total_bytes_read": 0,
        "errors": [],
        "status": "IN_PROGRESS",
    }
    if args.resume and partial.exists():
        resumed = json.loads(partial.read_text(encoding="utf-8"))
        if resumed["manifest_sha256"] != manifest_hash:
            raise SystemExit("Manifest changed; refusing unsafe resume")
        state = resumed
        state["resumed_at"] = utc_now()
    start_index = int(state["processed_files"])
    started = time.monotonic()

    for index, row in enumerate(rows[start_index:], start_index):
        relative = row["relative_path"].replace("\\", "/")
        try:
            path = Path(row["absolute_path"])
            payload = path.read_bytes()
            state["total_bytes_read"] += len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            if digest == row["sha256"]:
                state["sha256_match_files"] += 1
            else:
                state["errors"].append(
                    {"file": relative, "code": "SHA256_MISMATCH", "actual": digest}
                )
            with sf.SoundFile(io.BytesIO(payload)) as audio:
                values = audio.read(dtype="float32", always_2d=True)
                state["readable_files"] += 1
                if bool(np.isfinite(values).all()):
                    state["finite_files"] += 1
                else:
                    state["errors"].append({"file": relative, "code": "NON_FINITE_AUDIO"})
                metadata_matches = (
                    audio.samplerate == int(row["sample_rate"])
                    and audio.frames == int(row["num_frames"])
                    and audio.channels == int(row["num_channels"])
                    and len(payload) == int(row["file_size_bytes"])
                )
                if metadata_matches:
                    state["metadata_match_files"] += 1
                else:
                    state["errors"].append({"file": relative, "code": "METADATA_MISMATCH"})
        except Exception as exc:
            state["errors"].append(
                {"file": relative, "code": "READ_FAILURE", "error": type(exc).__name__}
            )
        state["processed_files"] = index + 1
        if state["processed_files"] % args.checkpoint_every == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            state["files_per_second_current_session"] = round(
                (state["processed_files"] - start_index) / elapsed, 3
            )
            state["checkpointed_at"] = utc_now()
            atomic_json(partial, state)

    expected = state["expected_files"]
    complete = all(
        state[key] == expected
        for key in (
            "processed_files", "readable_files", "finite_files",
            "sha256_match_files", "metadata_match_files",
        )
    )
    state["completed_at"] = utc_now()
    state["status"] = "VERIFIED" if complete and not state["errors"] else "FAILED_VERIFICATION"
    state["elapsed_seconds_current_session"] = round(time.monotonic() - started, 3)
    atomic_json(output, state)
    if partial.exists():
        partial.unlink()
    print(json.dumps({
        "output": output.name,
        "status": state["status"],
        "processed_files": state["processed_files"],
        "errors": len(state["errors"]),
    }, sort_keys=True))
    return 0 if state["status"] == "VERIFIED" else 7


if __name__ == "__main__":
    raise SystemExit(main())
