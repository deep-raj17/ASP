# ROS CLI Specification

Specification: ROS-FS-01  
Version: Draft 1.0

The CLI is an interface to core commands and queries. It never edits registries
or project files directly.

## Global conventions

Global options: `--project`, `--output text|json|yaml`, `--registry-position`,
`--policy`, `--dry-run`, `--non-interactive`, `--correlation-id`.

Mutating commands require an idempotency key (generated and displayed if
interactive), show the resolved plan, and honor approval policy. Structured
output uses stable schemas; diagnostics go to stderr.

## Commands

| Command | Purpose | Important arguments |
|---|---|---|
| `ros init` | Create project manifest candidate | `--name`, `--adapter`, `--path` |
| `ros status` | Show derived project/workflow state | `--workflow`, `--at` |
| `ros verify` | Run declared verifier | `--evidence`, `--method` |
| `ros run` | Start workflow or node | workflow, `--version`, `--idempotency-key` |
| `ros resume` | Resume blocked/failed run under policy | run ID, `--from` |
| `ros registry` | Query/export/verify registries | registry, `get|list|verify|export` |
| `ros evidence` | Register/query/verify evidence | `add|show|lineage|verify` |
| `ros workflow` | List/inspect/validate definitions | `list|show|validate|graph` |
| `ros gate` | Explain or reevaluate a gate | `show|explain|evaluate` |
| `ros project` | Validate/show project adapter | `validate|show|bindings` |
| `ros module` | Discover/resolve/inspect modules | `list|search|resolve|show` |
| `ros doctor` | Diagnose environment/integrity | `--scope`, `--fix` prohibited in FS-01 |
| `ros export` | Produce portable evidence bundle | `--at`, `--manifest` |
| `ros archive` | Request policy-controlled archival | target, approval reference |

`ros gate evaluate` requests deterministic evaluation; it cannot supply a
verdict. `ros evidence add` registers an observation; it cannot mark verified.

## Standard errors

| Code | Meaning |
|---|---|
| `ROS-E-SCHEMA` | Input failed schema validation |
| `ROS-E-CONFLICT` | Expected registry/aggregate version conflict |
| `ROS-E-POLICY` | Policy denied action |
| `ROS-E-APPROVAL` | Approval missing, expired, or out of scope |
| `ROS-E-BLOCKED` | Required dependency/evidence unavailable |
| `ROS-E-INTEGRITY` | Digest, signature, or chain failure |
| `ROS-E-COMPAT` | Version/capability incompatibility |
| `ROS-E-NOTFOUND` | Requested identity absent at selected position |

## Exit codes

`0` success; `2` usage/schema error; `3` blocked; `4` policy/approval denied;
`5` integrity failure; `6` execution failure; `7` compatibility failure;
`8` conflict; `10` partial/inconclusive verification; `20` internal error.

## Safety

Dry-run performs resolution and policy evaluation without dispatch or append.
JSON output never includes secrets. Destructive “fix” flags are absent from the
foundation CLI and require future separately specified commands.

