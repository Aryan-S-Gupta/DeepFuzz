#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from json2init.deepfuzz_common import load_json, write_json  # noqa: E402
from stage4.stage4_coverage_runner import (  # noqa: E402
    BUG_CSV_FIELDS,
    EXECUTION_SUMMARY_FIELDS,
    FAILURE_AUDIT_FIELDS,
    build_bug_report,
    build_coverage_report,
    build_execution_and_audit_rows,
    read_csv_rows,
    recompute_summary_from_merged_rows,
    write_bug_audit_csv,
    write_bug_report_csv,
    write_coverage_markdown,
    write_csv,
)


LOW_COVERAGE_FIELDS = [
    "api_full_name",
    "reason",
    "coverage_scope",
    "python_statement_percent",
    "python_branch_percent",
    "python_covered_lines",
    "python_coverable_lines",
    "native_line_percent",
    "native_covered_lines",
    "native_coverable_lines",
    "native_branch_percent",
    "native_function_percent",
    "python_json_path",
    "native_json_path",
    "python_export_error",
    "native_export_error",
    "worker_exit_code",
    "worker_timeout",
    "init_json_path",
    "results_json_path",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _low_coverage_rows(coverage_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = coverage_report.get("selected_function_coverage", {})
    python_json_path = selected.get("python_json_path", "") if isinstance(selected, dict) else ""
    rows: List[Dict[str, Any]] = []
    for item in coverage_report.get("low_coverage_apis", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "api_full_name": item.get("api_full_name", ""),
            "reason": item.get("reason", ""),
            "coverage_scope": "selected_function",
            "python_statement_percent": "",
            "python_branch_percent": "",
            "python_covered_lines": item.get("function_covered_lines", ""),
            "python_coverable_lines": item.get("function_coverable_lines", ""),
            "native_line_percent": "",
            "native_covered_lines": "",
            "native_coverable_lines": "",
            "native_branch_percent": "",
            "native_function_percent": "",
            "python_json_path": python_json_path,
            "native_json_path": "",
            "python_export_error": "",
            "native_export_error": "",
            "worker_exit_code": "",
            "worker_timeout": "",
            "init_json_path": "",
            "results_json_path": "",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh Stage 4 reports from existing worker artifacts without rerunning APIs")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "latest"))
    ap.add_argument("--stage4-results-dir", required=True)
    ap.add_argument("--known-bugs-file", default="")
    ap.add_argument("--unresolved-failures-csv", default="")
    args = ap.parse_args()

    results_dir = Path(args.stage4_results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    if not results_dir.exists():
        raise SystemExit(f"Stage 4 results dir does not exist: {results_dir}")

    summary_path = results_dir / "summary.json"
    summary = load_json(str(summary_path)) if summary_path.exists() else {}
    results_rows = read_csv_rows(results_dir / "results.csv")
    failure_rows = read_csv_rows(results_dir / "failures.csv")
    coverage_rows = read_csv_rows(results_dir / "coverage_report.csv")
    recompute_summary_from_merged_rows(summary, results_rows, failure_rows)

    mutation_budget = int(_float(summary.get("mutation_budget", 8), 8))
    elapsed = _float(summary.get("execution_time_seconds", 0.0), 0.0)
    bug_report = build_bug_report(
        str(results_dir),
        results_rows,
        mutation_budget=mutation_budget,
        known_bugs_file=args.known_bugs_file,
        unresolved_failures_file=args.unresolved_failures_csv,
    )
    coverage_report = build_coverage_report(
        summary=summary,
        coverage_rows=coverage_rows,
        results_rows=results_rows,
        bug_report=bug_report,
        execution_time_seconds=elapsed,
        native_requested=str(summary.get("native_coverage_engine", "none")) != "none",
        python_requested=bool(summary.get("python_coverage_enabled")),
        run_id=args.run_id,
    )
    summary["apis_with_low_coverage"] = coverage_report.get("apis_with_low_coverage", summary.get("apis_with_low_coverage", 0))
    summary["low_coverage_applicability"] = coverage_report.get("low_coverage_applicability", summary.get("low_coverage_applicability", ""))

    execution_rows, audit_rows = build_execution_and_audit_rows(results_rows)
    write_csv(results_dir / "execution_summary.csv", execution_rows, EXECUTION_SUMMARY_FIELDS)
    write_csv(results_dir / "failure.csv", audit_rows, FAILURE_AUDIT_FIELDS)
    write_csv(results_dir / "low_coverage.csv", _low_coverage_rows(coverage_report), LOW_COVERAGE_FIELDS)
    write_json(str(results_dir / "bug_report.json"), bug_report)
    write_bug_report_csv(results_dir / "bug_report.csv", bug_report)
    write_bug_audit_csv(results_dir / "bug_audit.csv", bug_report)
    write_json(str(results_dir / "coverage_report.json"), coverage_report)
    write_json(str(results_dir / "selected_function_coverage.json"), coverage_report.get("selected_function_coverage", {}))
    write_coverage_markdown(str(results_dir / "coverage_report.md"), coverage_report, bug_report)
    write_json(str(summary_path), summary)

    print(f"[refresh] candidate_bugs={bug_report.get('summary', {}).get('total_candidates', 0)}")
    print(f"[refresh] bug_csv_rows={len(bug_report.get('candidate_bugs', []) or [])}")
    print(f"[refresh] bug_audit_csv={results_dir / 'bug_audit.csv'}")
    print(f"[refresh] apis_with_low_coverage={summary.get('apis_with_low_coverage')}")
    print(f"[refresh] wrote {results_dir / 'coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
