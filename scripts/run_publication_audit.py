"""
scripts/run_publication_audit.py
────────────────────────────────────────────────────────
Independent "Go/No-Go" Publication Audit for CHAAD Project.

This script performs ALL checks required before submission and
produces a single pass/fail verdict with detailed reasoning.

Decision gates (each must pass for GO):
  1.  Data Leakage Audit          – No cross-split leakage
  2.  Reproducibility             – Dependencies frozen, seeds controlled
  3.  Shortcut Learning           – Model doesn't exploit trivial features
  4.  Baseline Comparisons        – Compared against strong baselines
  5.  Ablation Studies            – Each component's contribution isolated
  6.  Statistical Validation      – CIs, significance tests reported
  7.  Subgroup Analysis           – Performance across subpopulations
  8.  Robustness Analysis         – Failure modes characterized
  9.  Novelty Defined             – Clear research contribution stated
  10. Manuscript Ready            – All figures, tables, references prepared

Usage:
    python scripts/run_publication_audit.py [--output reports/go_nogo_report.json]
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.split_utils import get_repo_commit

# ─────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class AuditGate:
    gate_id: str
    name: str
    description: str
    status: str             # "PASS", "FAIL", "WARNING", "NOT_CHECKED"
    evidence: List[str]     # File paths or citations
    recommendation: str     # What to do if failing
    weight: float = 1.0     # Importance weight for overall score

    def is_pass(self) -> bool:
        return self.status == "PASS"


@dataclass
class AuditReport:
    timestamp: str
    git_commit: Optional[str]
    project_root: str
    gates: List[AuditGate]
    overall_verdict: str       # "GO", "NO-GO", "CONDITIONAL"
    overall_score: float       # 0-100
    passed_gates: int
    total_gates: int
    critical_failures: List[str]
    warnings: List[str]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "project_root": self.project_root,
            "gates": [asdict(g) for g in self.gates],
            "overall_verdict": self.overall_verdict,
            "overall_score": self.overall_score,
            "passed_gates": self.passed_gates,
            "total_gates": self.total_gates,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


# ─────────────────────────────────────────────────────────
#  Check Implementations
# ─────────────────────────────────────────────────────────

def check_gate_a_data_leakage() -> AuditGate:
    """Gate A: Verify no data leakage in splits."""
    evidence = []
    status = "PASS"
    rec = ""

    # Check 1: Machine split table
    table_path = Path("artifacts/research_audit/machine_split_table.csv")
    if table_path.exists():
        df = pd.read_csv(table_path)
        # Verify each machine_id appears in exactly one split
        mid_splits = df.groupby("machine_id")["split"].nunique()
        if (mid_splits > 1).any():
            status = "FAIL"
            rec = "Machine IDs appear in multiple splits. Regenerate splits with machine-independent protocol."
        else:
            evidence.append(str(table_path))
    else:
        evidence.append("machine_split_table.csv NOT FOUND")
        status = "FAIL"
        rec = "Run _audit_check.py to generate machine split table."

    # Check 2: SHA-256 duplicate report
    dup_path = Path("artifacts/research_audit/duplicate_hash_report.csv")
    if dup_path.exists():
        evidence.append(str(dup_path))
    else:
        evidence.append("duplicate_hash_report.csv NOT FOUND")

    # Check 3: Research integrity report
    integrity_path = Path("reports/research_integrity_report.json")
    if integrity_path.exists():
        with open(integrity_path) as f:
            report = json.load(f)
        failed_gates = report.get("scores", {}).get("gates_failed", 0)
        if failed_gates > 0:
            status = "FAIL"
            rec = f"Research integrity report shows {failed_gates} failed gates."
        evidence.append(str(integrity_path))
    else:
        evidence.append("research_integrity_report.json NOT FOUND")

    return AuditGate(
        gate_id="A",
        name="Data Leakage Audit",
        description="Verify no cross-split leakage: machine ID isolation, SHA-256 duplicates, segment overlap.",
        status=status,
        evidence=evidence,
        recommendation=rec or "No action needed.",
        weight=2.0,  # Critical
    )


def check_gate_b_reproducibility() -> AuditGate:
    """Gate B: Verify reproducibility measures are in place."""
    evidence = []
    status = "PASS"
    issues = []

    # Check seed utility
    seed_path = Path("utils/seed.py")
    if seed_path.exists():
        evidence.append(str(seed_path))
    else:
        issues.append("utils/seed.py missing")

    # Check config has seed
    if Path("config.py").exists():
        content = Path("config.py").read_text()
        if "random_seed" in content:
            evidence.append("config.py contains random_seed")
        else:
            issues.append("config.py missing random_seed")
    else:
        issues.append("config.py missing")

    # Check provenance
    prov_path = Path("artifacts/experiment_provenance.json")
    if prov_path.exists():
        evidence.append(str(prov_path))
    else:
        issues.append("experiment_provenance.json not recorded yet (train once)")

    # Check requirements lock
    if Path("requirements.txt").exists():
        evidence.append("requirements.txt exists")
    else:
        issues.append("requirements.txt missing")

    if issues:
        status = "FAIL" if len(issues) > 1 else "WARNING"

    return AuditGate(
        gate_id="B",
        name="Reproducibility",
        description="Seeds controlled, dependencies frozen, provenance tracked.",
        status=status,
        evidence=evidence,
        recommendation="; ".join(issues) if issues else "No action needed.",
        weight=1.5,
    )


def check_gate_c_shortcut_learning() -> AuditGate:
    """Gate C: Verify model doesn't exploit trivial shortcuts."""
    evidence = []
    status = "NOT_CHECKED"
    rec = ""

    # Check for shortcut audit report
    shortcut_path = Path("artifacts/research_audit/shortcut_learning_report.json")
    if shortcut_path.exists():
        with open(shortcut_path) as f:
            report = json.load(f)
        overd = report.get("overall", {})
        if overd.get("shortcut_detected", True):
            status = "FAIL"
            rec = "Shortcut learning detected. Model may exploit trivial features."
        else:
            status = "PASS"
            rec = "No shortcut learning detected."
        evidence.append(str(shortcut_path))
    else:
        # Check if audit was run
        script_path = Path("scripts/audit_shortcuts.py")
        if script_path.exists():
            rec = "Run: python scripts/audit_shortcuts.py"
        else:
            rec = "Create and run shortcut learning audit script."
        evidence.append("shortcut_learning_report.json NOT FOUND")

    return AuditGate(
        gate_id="C",
        name="Shortcut Learning",
        description="Model does not exploit metadata-only features or trivial artifacts.",
        status=status,
        evidence=evidence,
        recommendation=rec,
        weight=1.5,
    )


def check_gate_d_baselines() -> AuditGate:
    """Gate D: Verify baseline comparisons exist."""
    evidence = []
    status = "NOT_CHECKED"

    report_path = Path("reports/baseline_comparison.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        results = report.get("results", [])
        n_baselines = len(results)
        if n_baselines >= 8:
            status = "PASS"
        elif n_baselines >= 4:
            status = "WARNING"
        else:
            status = "FAIL"
        evidence.append(f"{str(report_path)} ({n_baselines} baselines)")
    else:
        evidence.append("baseline_comparison.json NOT FOUND")

    return AuditGate(
        gate_id="D",
        name="Baseline Comparisons",
        description="Strong baselines reported: random, class-prior, single-score, fixed-fusion, classical ML.",
        status=status,
        evidence=evidence,
        recommendation="Run: python scripts/run_baselines.py" if status != "PASS" else "No action needed.",
        weight=1.5,
    )


def check_gate_e_ablations() -> AuditGate:
    """Gate E: Verify ablation studies are documented."""
    evidence = []
    status = "NOT_CHECKED"

    # Check novelty doc
    novelty_path = Path("docs/NOVELTY_AND_CONTRIBUTIONS.md")
    if novelty_path.exists():
        content = novelty_path.read_text(encoding="utf-8")
        if "Ablation" in content or "ablation" in content.lower():
            evidence.append(str(novelty_path))
        else:
            evidence.append("NOVELTY_AND_CONTRIBUTIONS.md missing ablation plan")

    # Check for ablation report
    ablation_path = Path("reports/ablation_study_report.json")
    if ablation_path.exists():
        evidence.append(str(ablation_path))
        status = "PASS"
    else:
        evidence.append("ablation_study_report.json NOT FOUND")

    return AuditGate(
        gate_id="E",
        name="Ablation Studies",
        description="Each component's contribution isolated: reliability gate, condition-awareness, score sources.",
        status=status,
        evidence=evidence,
        recommendation="Run ablation studies and generate report." if status != "PASS" else "No action needed.",
        weight=1.0,
    )


def check_gate_f_statistics() -> AuditGate:
    """Gate F: Verify statistical validation is reported."""
    evidence = []
    status = "NOT_CHECKED"

    report_path = Path("reports/statistical_validation_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        evidence.append(str(report_path))

        if "bootstrap_confidence_intervals" in report and "delong_test" in report:
            status = "PASS"
        else:
            status = "WARNING"
    else:
        evidence.append("statistical_validation_report.json NOT FOUND")

    return AuditGate(
        gate_id="F",
        name="Statistical Validation",
        description="Bootstrap CIs, significance tests (DeLong, McNemar, Wilcoxon), effect sizes.",
        status=status,
        evidence=evidence,
        recommendation="Run: python scripts/statistical_validation.py" if status != "PASS" else "No action needed.",
        weight=1.0,
    )


def check_gate_g_subgroup() -> AuditGate:
    """Gate G: Verify subgroup analysis."""
    evidence = []
    status = "NOT_CHECKED"

    reports = [
        "reports/per_machine_results.csv",
        "reports/per_machine_id_results.csv",
        "reports/per_noise_condition_results.csv",
    ]

    found = 0
    for r in reports:
        if Path(r).exists():
            evidence.append(r)
            found += 1
        else:
            evidence.append(f"{r} NOT FOUND")

    if found >= 3:
        status = "PASS"
    elif found >= 1:
        status = "WARNING"
    else:
        status = "FAIL"

    return AuditGate(
        gate_id="G",
        name="Subgroup Analysis",
        description="Performance reported per machine type, machine ID, and noise condition.",
        status=status,
        evidence=evidence,
        recommendation="Run: python scripts/evaluate_subgroups.py" if status != "PASS" else "No action needed.",
        weight=1.0,
    )


def check_gate_h_robustness() -> AuditGate:
    """Gate H: Verify robustness and failure analysis."""
    evidence = []
    status = "NOT_CHECKED"

    # Check for robustness report
    rob_path = Path("reports/robustness_report.json")
    if rob_path.exists():
        evidence.append(str(rob_path))
        status = "PASS"
    else:
        evidence.append("robustness_report.json NOT FOUND")

    # Check for limitations doc
    lim_path = Path("docs/LIMITATIONS.md")
    if lim_path.exists():
        evidence.append(str(lim_path))
    else:
        evidence.append("LIMITATIONS.md NOT FOUND")

    return AuditGate(
        gate_id="H",
        name="Robustness & Failure Analysis",
        description="Failure modes characterized, confusion analysis, calibration errors, perturbations.",
        status=status,
        evidence=evidence,
        recommendation="Create robustness analysis report." if status != "PASS" else "No action needed.",
        weight=1.0,
    )


def check_gate_i_novelty() -> AuditGate:
    """Gate I: Verify research contribution is clearly defined."""
    evidence = []
    status = "NOT_CHECKED"

    novelty_path = Path("docs/NOVELTY_AND_CONTRIBUTIONS.md")
    if novelty_path.exists():
        content = novelty_path.read_text(encoding="utf-8")
        evidence.append(str(novelty_path))

        # Check for key sections
        has_problem = "research problem" in content.lower()
        has_gap = "research gap" in content.lower()
        has_method = "mathematical" in content.lower() or "formulation" in content.lower()
        has_contrast = "differs from" in content.lower() or "different from" in content.lower()

        if has_problem and has_gap and has_method and has_contrast:
            status = "PASS"
        elif has_problem and has_gap:
            status = "WARNING"
        else:
            status = "FAIL"
    else:
        evidence.append("NOVELTY_AND_CONTRIBUTIONS.md NOT FOUND")
        status = "FAIL"

    # Check code implementation of novel contribution
    reliability_path = Path("models/reliability.py")
    if reliability_path.exists():
        evidence.append(str(reliability_path))
        if status != "PASS":
            status = "WARNING"  # Code exists but needs doc completion
    else:
        evidence.append("models/reliability.py NOT FOUND")

    return AuditGate(
        gate_id="I",
        name="Novelty Defined",
        description="Clear research gap, mathematical formulation, novel contribution isolated from engineering.",
        status=status,
        evidence=evidence,
        recommendation="Complete NOVELTY_AND_CONTRIBUTIONS.md with clear research gap and formulation." if status != "PASS" else "No action needed.",
        weight=2.0,  # Critical
    )


def check_gate_j_manuscript() -> AuditGate:
    """Gate J: Verify manuscript materials are prepared."""
    evidence = []
    status = "NOT_CHECKED"

    # Check for evaluation report
    eval_path = Path("reports/final_evaluation_report.json")
    if eval_path.exists():
        evidence.append(str(eval_path))
        status = "WARNING"  # Report exists but manuscript not yet generated
    else:
        evidence.append("final_evaluation_report.json NOT FOUND")

    manuscript_path = Path("reports/manuscript.md")
    if manuscript_path.exists():
        evidence.append(str(manuscript_path))
        status = "PASS"
    else:
        evidence.append("manuscript.md NOT FOUND")

    return AuditGate(
        gate_id="J",
        name="Manuscript Ready",
        description="Complete manuscript with all figures, tables, references, and supplementary materials.",
        status=status,
        evidence=evidence,
        recommendation="Generate manuscript from evaluation results." if status != "PASS" else "No action needed.",
        weight=1.5,
    )


# ─────────────────────────────────────────────────────────
#  Main Audit Runner
# ─────────────────────────────────────────────────────────

def run_full_audit() -> AuditReport:
    """Execute all 10 publication gates and produce verdict."""

    project_root = str(Path(__file__).resolve().parent.parent)

    gates: List[AuditGate] = [
        check_gate_a_data_leakage(),
        check_gate_b_reproducibility(),
        check_gate_c_shortcut_learning(),
        check_gate_d_baselines(),
        check_gate_e_ablations(),
        check_gate_f_statistics(),
        check_gate_g_subgroup(),
        check_gate_h_robustness(),
        check_gate_i_novelty(),
        check_gate_j_manuscript(),
    ]

    # Compute scores
    total_weight = sum(g.weight for g in gates)
    passed_weight = sum(g.weight for g in gates if g.is_pass())
    failed_weight = sum(g.weight for g in gates if g.status == "FAIL")
    not_checked_weight = sum(g.weight for g in gates if g.status == "NOT_CHECKED")

    overall_score = (passed_weight / total_weight) * 100 if total_weight > 0 else 0
    passed_gates = sum(1 for g in gates if g.is_pass())
    total_gates = len(gates)

    critical_failures = [g.name for g in gates if g.status == "FAIL" and g.weight >= 1.5]
    warnings = [g.name for g in gates if g.status == "WARNING"]

    # Verdict logic
    nogo_threshold = any(g.status == "FAIL" for g in gates if g.weight >= 2.0)
    conditional = overall_score >= 70 and not nogo_threshold

    if overall_score >= 90 and not nogo_threshold and len(critical_failures) == 0:
        verdict = "GO"
        recommendation = (
            "Project meets all publication standards. Proceed with submission.\n"
            "Review warnings above and ensure all supplementary materials are compiled."
        )
    elif nogo_threshold:
        verdict = "NO-GO"
        recommendation = (
            "Critical failures detected. Address these before resubmission:\n"
            + "\n".join(f"  - {f}" for f in critical_failures)
        )
    elif conditional:
        verdict = "CONDITIONAL"
        recommendation = (
            "Project is close to publication-ready but has warnings/not-checked gates.\n"
            f"Address {len(warnings)} warnings and {sum(1 for g in gates if g.status == 'NOT_CHECKED')} unchecked gates."
        )
    else:
        verdict = "NO-GO"
        recommendation = (
            "Too many gates failing or not checked. Complete the pipeline before submission."
        )

    return AuditReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_repo_commit(),
        project_root=project_root,
        gates=gates,
        overall_verdict=verdict,
        overall_score=round(overall_score, 1),
        passed_gates=passed_gates,
        total_gates=total_gates,
        critical_failures=critical_failures,
        warnings=warnings,
        recommendation=recommendation,
    )


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CHAAD Publication Go/No-Go Audit")
    parser.add_argument("--output", default="reports/go_nogo_report.json", help="Output path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    print("=" * 70)
    print("  CHAAD PUBLICATION AUDIT - GO/NO-GO VERDICT")
    print("=" * 70)

    report = run_full_audit()

    # Print summary
    print(f"\n  Verdict: {report.overall_verdict}")
    print(f"  Score:   {report.overall_score:.1f}%")
    print(f"  Gates:   {report.passed_gates}/{report.total_gates} passed")
    print(f"  Critical Failures: {len(report.critical_failures)}")
    print(f"  Warnings: {len(report.warnings)}")

    print(f"\n{'─'*70}")
    print(f"  {'Gate':<6s} {'Name':<35s} {'Status':<15s}")
    print(f"{'─'*70}")
    for g in report.gates:
        icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "NOT_CHECKED": "❓"}.get(g.status, "?")
        print(f"  {g.gate_id:<6s} {g.name:<35s} {g.status:<15s} {icon}")
    print(f"{'─'*70}")

    if args.verbose:
        print(f"\n  Recommendation:\n  {report.recommendation}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n  Report saved to: {output_path}")

    # Exit code based on verdict
    if report.overall_verdict == "NO-GO":
        sys.exit(1)
    elif report.overall_verdict == "CONDITIONAL":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
