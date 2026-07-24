# ROS Agent Governance

Specification: ROS-FS-01  
Version: Draft 1.0

This file specifies ROS agents; it is not an implementation instruction file
for the surrounding CHAAD repository.

## Agent types

Research, Coding, Reviewer, Publication, Statistics, Documentation, Deployment,
and Project Management agents are task-specialized executors.

## Mandatory constraints

Agents:

- MUST act only on a delegated task with bounded scope and permissions;
- MUST identify generated outputs, inputs, assumptions, and uncertainties;
- MUST submit candidate artifacts through normal evidence collection;
- MUST preserve failed attempts and negative results;
- MUST stop when policy, approval, evidence, or project boundaries block work;
- MUST NOT approve gates, approvals, submissions, or releases;
- MUST NOT alter evidence, registry history, external reviewer records, or
  scientific results;
- MUST NOT broaden delegation or infer high-risk authority;
- MUST NOT represent generated prose as verification.

## Delegation contract

A task delegation includes task ID, actor, project, permitted actions and
resources, prohibited actions, input artifact digests, expected output schemas,
deadline/budget, required approvals, and idempotency key. Subdelegation is
allowed only when explicitly permitted and cannot expand authority.

## Execution record

Each attempt records agent identity/version, provider/model or executable,
prompt/task digest where applicable, tool calls, environment, outputs, errors,
time, token/compute usage where available, and parent delegation.

## Agent-produced material

Code, reports, analyses, and manuscripts are candidate artifacts. Deterministic
checks, independent verifiers, policy, and human review determine their use.
Reviewer simulation is labeled `simulated` and cannot create a `ReviewReceived`
external event.

## Conflict handling

Agents follow the highest-priority applicable policy and explicit user
authority. Conflicting or ambiguous instructions produce a blocked task and an
audit explanation. Agents never resolve scientific conflicts by silently
choosing the most convenient evidence.

## Separation of duties

For high-risk actions, the executor cannot be the sole verifier or approver.
Agent identity never satisfies a human-approval role.

