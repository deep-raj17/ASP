# CHAAD Adapter Specification

Adapter version 1.0.0 binds project `chaad` to PMPS workflow 1.0.0. All public
configuration uses repository-relative paths or environment-variable
placeholders. Private runtime state lives under ignored `.ros/`.

The adapter is declarative except for `import/import_chaad.py`, which calls the
public ROS service, evidence engine, registry contract, and workflow engine.
It contains no CHAAD behavior in ROS core. Imports are append-only, first
validated in an ephemeral workspace, and idempotent by deterministic record and
evidence IDs.

Scientific artifacts are inputs only. Invalid original and corrected
validation exports have separate identities and a supersession link. Dataset,
experiment, model, publication, and workflow records use incomplete,
provisional, invalid, or blocked states where evidence demands them.
