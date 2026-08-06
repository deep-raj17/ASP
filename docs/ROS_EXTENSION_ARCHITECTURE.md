# ROS Extension Architecture

This document records the proposed post-ROS-DATA architecture. It is a
roadmap, not an implementation certificate.

## Domains

- **ROS-ML:** machine-learning lifecycle governance
- **RV:** research validation (protocol, leakage, splits, reproducibility,
  baselines, ablations, statistics, errors, robustness, explainability)
- **RW:** manuscript authoring and presentation
- **RR:** independent review simulation and revision management
- **RP:** submission, artifact evaluation, camera-ready, and lifecycle

## Recommended order

1. ROS-ML
2. RV
3. RW
4. RR
5. RP

## Gate policy

These are proposed future frameworks. They must not be marked complete merely
because their specifications exist. Each framework needs an explicit
prerequisite definition, machine-readable evidence, validation outputs, and a
PASS/BLOCKED gate. The current repository state remains authoritative:
ROS-PROJECT, ROS-PUB, ROS-DEPLOY, ROS-SEC, and ROS-DATA are not complete.

No RV, RW, RR, or RP stage is started by this roadmap entry.
