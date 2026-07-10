#!/usr/bin/env python3
"""Generate FINORA quarterly research change-control review.

This report protects the frozen baseline. It can approve only evidence-driven
dataset improvements, bug fixes, or narrow research-engineering changes. It
never approves broker APIs, real-money trading, new models, new frameworks, or
architecture expansion.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.daily_research_run import DISALLOWED_EXECUTION_TERMS  # noqa: E402
from scripts.generate_paper_review_report import markdown_list  # noqa: E402
from scripts.generate_weekly_research_review import markdown_table, read_markdown_records, read_text_if_exists  # noqa: E402
from scripts.update_paper_observations import read_csv_rows  # noqa: E402

FORBIDDEN_CHANGE_TERMS = {
    "broker",
    "real-money",
    "real money",
    "order",
    "execution",
    "new model",
    "new framework",
    "architecture",
}
REPRODUCIBILITY_TERMS = {"reproducibility", "correctness", "data quality", "research capability", "dataset", "bug", "audit"}


def assert_research_only() -> None:
    imported = {name.lower() for name in sys.modules}
    disallowed = [term for term in DISALLOWED_EXECUTION_TERMS if any(term in module for module in imported)]
    if disallowed:
        raise RuntimeError(f"Real-trading module detected in process: {', '.join(disallowed)}")


def parse_run_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else datetime.now(timezone.utc).date()


def quarter_label(run_date: date) -> str:
    quarter = ((run_date.month - 1) // 3) + 1
    return f"{run_date.year:04d}-Q{quarter}"


def row_date(row: dict[str, str]) -> date | None:
    try:
        return date.fromisoformat(row.get("date", ""))
    except ValueError:
        return None


def rows_for_quarter(rows: list[dict[str, str]], run_date: date) -> list[dict[str, str]]:
    quarter = ((run_date.month - 1) // 3) + 1
    selected: list[dict[str, str]] = []
    for row in rows:
        parsed = row_date(row)
        if parsed is None:
            continue
        row_quarter = ((parsed.month - 1) // 3) + 1
        if parsed.year == run_date.year and row_quarter == quarter:
            selected.append(row)
    return selected


def read_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
        elif path.is_dir():
            texts.extend(read_markdown_records(path))
    return texts


def load_evidence_text(root: Path) -> str:
    evidence_parts = []
    evidence_parts.extend(read_markdown_records(root / "reports" / "monthly"))
    evidence_parts.extend(read_markdown_records(root / "reports" / "weekly"))
    evidence_parts.extend(read_markdown_records(root / "reports" / "research" / "knowledge_base"))
    evidence_parts.extend(read_markdown_records(root / "reports" / "research" / "decision_reviews"))
    evidence_parts.extend(read_markdown_records(root / "docs" / "research_change_control"))
    evidence_parts.append(read_text_if_exists(root / "reports" / "meta_research_dashboard.md"))
    return "\n".join(part for part in evidence_parts if part)


def load_proposed_changes(root: Path) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for path in (
        root / "reports" / "change_control" / "proposed_changes.csv",
        root / "docs" / "research_change_control" / "proposed_changes.csv",
    ):
        candidates.extend(read_csv_rows(path))
    return candidates


def contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        if " " in term or "-" in term:
            if term in lowered:
                return True
            continue
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return True
    return False


def evidence_mentions(change: dict[str, str], evidence_text: str) -> bool:
    haystack = evidence_text.lower()
    identifiers = [
        change.get("change_id", ""),
        change.get("title", ""),
        change.get("documented_limitation", ""),
        change.get("evidence_refs", ""),
    ]
    return any(identifier and identifier.lower() in haystack for identifier in identifiers)


def classify_change(change: dict[str, str], evidence_text: str) -> dict[str, str]:
    title = change.get("title", "Untitled change")
    change_type = change.get("change_type", "").lower()
    combined = " ".join(str(value) for value in change.values())
    documented = bool(change.get("documented_limitation", "").strip())
    appears_in_evidence = evidence_mentions(change, evidence_text)
    improves_research = contains_any(combined, REPRODUCIBILITY_TERMS)
    novelty_only = change.get("novelty_risk", "").lower() in {"true", "yes", "1"} or "novelty" in combined.lower()
    forbidden = contains_any(combined, FORBIDDEN_CHANGE_TERMS)

    if forbidden:
        decision = "reject_change_request"
        reason = "forbidden expansion or trading-related term detected"
    elif not documented or not appears_in_evidence:
        decision = "defer_until_more_evidence"
        reason = "missing documented limitation or monthly/weekly evidence"
    elif not improves_research or novelty_only:
        decision = "reject_change_request"
        reason = "does not clearly improve reproducibility, correctness, data quality, or research capability"
    elif "dataset" in change_type or "data" in change_type:
        decision = "dataset_improvement"
        reason = "evidence ties change to data-quality or dataset limitation"
    elif "bug" in change_type:
        decision = "bug_fix"
        reason = "evidence ties change to correctness or reproducibility bug"
    elif "engineering" in change_type or "infra" in change_type:
        decision = "evidence_driven_engineering"
        reason = "evidence ties change to research capability or reproducibility"
    elif change_type in {"none", "no_change"}:
        decision = "no_change"
        reason = "proposal records no implementation change"
    else:
        decision = "defer_until_more_evidence"
        reason = "change type is not specific enough for quarterly approval"

    return {
        "change_id": change.get("change_id", title),
        "title": title,
        "decision": decision,
        "reason": reason,
        "risk_of_changing": change.get("risk_of_changing", "review required before implementation"),
        "risk_of_not_changing": change.get("risk_of_not_changing", "research limitation may persist"),
    }


def count_text_occurrences(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def quarterly_metrics(root: Path, run_date: date, evidence_text: str) -> dict[str, int]:
    paper_rows = read_csv_rows(root / "reports" / "paper_trading" / "paper_trading_log.csv")
    rows = rows_for_quarter(paper_rows, run_date)
    metric_rows = rows if rows else paper_rows
    monthly_reports = read_markdown_records(root / "reports" / "monthly")
    hypotheses = {row.get("hypothesis_id") or f"{row.get('model', '')}:{row.get('horizon', '')}" for row in metric_rows}
    return {
        "research_campaigns_run": len(monthly_reports),
        "hypotheses_reviewed": len({item for item in hypotheses if item.strip()}),
        "hypotheses_rejected": sum(1 for row in metric_rows if row.get("current_status") == "rejected_after_review"),
        "hypotheses_continued": sum(1 for row in metric_rows if row.get("current_status") in {"reviewed", "completed", "active", "opened"}),
        "hypotheses_promoted": sum(1 for row in metric_rows if row.get("current_status") == "promoted_to_deeper_research"),
        "repeated_data_quality_failures": sum(1 for row in metric_rows if row.get("data_quality_warning") or row.get("current_status") == "expired"),
        "repeated_model_instability_issues": count_text_occurrences(evidence_text, {"model instability", "unstable", "disagreement"}),
        "recurring_feature_limitations": count_text_occurrences(evidence_text, {"weak feature", "feature limitation", "feature drift"}),
        "infrastructure_limitations": count_text_occurrences(evidence_text, {"infrastructure", "runtime", "storage", "latency"}),
        "unresolved_reproducibility_bugs": count_text_occurrences(evidence_text, {"unresolved bug", "reproducibility bug", "not reproducible"}),
    }


def split_decisions(decisions: list[dict[str, str]], labels: set[str]) -> list[str]:
    return [f"{row['change_id']}: {row['title']} ({row['reason']})" for row in decisions if row["decision"] in labels]


def generate_quarterly_change_control_review(root: Path | str = ".", run_date: date | None = None) -> dict[str, Any]:
    assert_research_only()
    root_path = Path(root).resolve()
    as_of = run_date or datetime.now(timezone.utc).date()
    label = quarter_label(as_of)
    evidence_text = load_evidence_text(root_path)
    proposed_changes = load_proposed_changes(root_path)
    decisions = [classify_change(change, evidence_text) for change in proposed_changes]
    metrics = quarterly_metrics(root_path, as_of, evidence_text)
    approved = split_decisions(decisions, {"dataset_improvement", "bug_fix", "evidence_driven_engineering"})
    rejected = split_decisions(decisions, {"reject_change_request"})
    deferred = split_decisions(decisions, {"defer_until_more_evidence", "no_change"})

    content = f"""# FINORA Quarterly Change-Control Review - {label}

Research change-control only. No broker APIs were called, no orders were submitted, and no real-money trading path is approved.

## Executive Summary
This review protects the frozen engineering baseline by allowing only evidence-driven dataset improvements, bug fixes, or narrow research-engineering changes.
Proposed changes reviewed: {len(decisions)}.
Approved changes: {len(approved)}.

## Quarterly Review Metrics
- Research campaigns run: {metrics["research_campaigns_run"]}
- Hypotheses reviewed: {metrics["hypotheses_reviewed"]}
- Hypotheses rejected: {metrics["hypotheses_rejected"]}
- Hypotheses continued: {metrics["hypotheses_continued"]}
- Hypotheses promoted to robustness review: {metrics["hypotheses_promoted"]}
- Repeated data-quality failures: {metrics["repeated_data_quality_failures"]}
- Repeated model instability issues: {metrics["repeated_model_instability_issues"]}
- Recurring feature limitations: {metrics["recurring_feature_limitations"]}
- Infrastructure limitations discovered by research: {metrics["infrastructure_limitations"]}
- Unresolved bugs affecting reproducibility: {metrics["unresolved_reproducibility_bugs"]}

## Research Evidence Summary
Evidence sources loaded from monthly governance reports, weekly reviews, research knowledge base,
meta-research dashboard, change-control docs, paper logs, and decision review records.

## Change-Control Decision Table
{markdown_table(decisions, ["change_id", "title", "decision", "reason", "risk_of_changing", "risk_of_not_changing"])}

## Approved Changes
{markdown_list(approved)}

## Rejected Changes
{markdown_list(rejected)}

## Deferred Changes
{markdown_list(deferred)}

## Evidence Requirements
No engineering change may be approved unless it is tied to a documented research limitation,
appears in monthly or weekly evidence, improves reproducibility, correctness, data quality,
or research capability, and does not add novelty for its own sake.

## Risk of Changing
Approved changes still require scoped implementation review, tests, and confirmation that they do not
introduce broker APIs, new models, new frameworks, or architecture expansion.

## Risk of Not Changing
Rejected or deferred limitations may continue to affect research reproducibility, correctness,
data quality, or research capability until better evidence exists.

## Next-Quarter Research Priorities
- Resolve approved data-quality, bug, and reproducibility limitations.
- Keep deferred changes under observation until evidence is stronger.
- Reject novelty-only requests that do not protect the frozen baseline.

## Boundary
This report is not feature expansion, not trading approval, and not investment advice. Real-money trading remains out of scope.
"""
    quarterly_dir = root_path / "reports" / "quarterly"
    quarterly_dir.mkdir(parents=True, exist_ok=True)
    summary_path = quarterly_dir / f"{label}_quarterly_change_control_review.md"
    summary_path.write_text(content, encoding="utf-8")
    assert_research_only()
    return {
        "summary_path": summary_path,
        "proposed_changes": len(decisions),
        "approved_changes": len(approved),
        "rejected_changes": len(rejected),
        "deferred_changes": len(deferred),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FINORA quarterly research change-control review.")
    parser.add_argument("--root", default=".", help="FINORA workspace root")
    parser.add_argument("--date", help="Review date in YYYY-MM-DD format")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_quarterly_change_control_review(args.root, run_date=parse_run_date(args.date))
    print("FINORA quarterly change-control review generated.")
    print(f"Quarterly review: {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
