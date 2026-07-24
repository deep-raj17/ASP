from pathlib import Path

import pytest

from ros.core.workflow_engine.errors import ErrorCode, WorkflowError
from ros.core.workflow_engine.loader import load_workflow


def write(tmp_path, text):
    path = tmp_path / "workflow.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_rejects_unsafe_or_unsupported_schema(tmp_path):
    path = write(tmp_path, "schema_version: unknown\nid: x\nversion: 1.0.0\ngates: []\n")
    with pytest.raises(WorkflowError) as exc:
        load_workflow(path)
    assert exc.value.code is ErrorCode.UNSUPPORTED_SCHEMA_VERSION


def test_rejects_cycle(tmp_path):
    path = write(
        tmp_path,
        """
schema_version: ros.workflow/v1
id: cyclic
version: 1.0.0
gates:
  - {id: a, title: A, entry: true, prerequisites: [b]}
  - {id: b, title: B, terminal: true, prerequisites: [a]}
""",
    )
    with pytest.raises(WorkflowError) as exc:
        load_workflow(path)
    assert exc.value.code is ErrorCode.CYCLIC_DEPENDENCY


def test_rejects_duplicate_and_bad_version(tmp_path):
    path = write(
        tmp_path,
        """
schema_version: ros.workflow/v1
id: duplicate
version: 1.0
gates:
  - {id: a, title: A, entry: true}
  - {id: a, title: B, terminal: true}
""",
    )
    with pytest.raises(WorkflowError) as exc:
        load_workflow(path)
    assert exc.value.code is ErrorCode.INVALID_WORKFLOW_DEFINITION
