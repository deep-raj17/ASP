# ROS-CORE-01 Repository Audit

The repository contained approved ROS-FS-01 Markdown specifications but no ROS
runtime package, workflow schema implementation, tests, or dependency metadata.
CHAAD occupied the existing root and had extensive unrelated uncommitted work.
ROS was therefore isolated under `ros/`, `ros/specs/`, and `tests/ros`; commits
are path-scoped. Python 3.10, PyYAML, and pytest were available. mypy, Ruff, and
Black were unavailable and were not installed.

The specification used `DRAFT/EVALUATING` and disallowed waiver while
ROS-CORE-01 required `NOT_STARTED/PENDING` and policy-authorized `WAIVED`.
ROS-FS-01 Draft 1.0.1 records the resolved vocabulary and preserves the rule
that waiver is not scientific satisfaction.
