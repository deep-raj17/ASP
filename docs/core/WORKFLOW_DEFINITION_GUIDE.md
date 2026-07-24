# Workflow Definition Guide

Workflow YAML declares `schema_version`, stable `id`, SemVer `version`, gates,
prerequisites, entry/terminal markers, retry limits, parallel groups, and
waiver policy. Loading uses `yaml.safe_load`; malformed definitions are rejected
without repair. See `ros/specs/workflows/research-validation-demo.yaml`.
