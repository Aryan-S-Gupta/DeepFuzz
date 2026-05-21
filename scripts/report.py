#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api_policy import is_internal_api
from common.result_io import atomic_write_csv, atomic_write_json, atomic_write_text


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def count_csv(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def csv_stage_summary(stage: str, ok_csv: Path, errors_csv: Path) -> Dict[str, Any]:
    ok = count_csv(ok_csv)
    errors = count_csv(errors_csv)
    return {
        "schema_version": "3.0",
        "stage": stage,
        "ok_csv": str(ok_csv),
        "errors_csv": str(errors_csv),
        "total_valid": ok,
        "total_failed": errors,
        "total_seen": ok + errors,
    }


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], preferred_fields: Sequence[str] | None = None) -> None:
    fields: List[str] = list(preferred_fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["api_full_name", "classification", "stage", "error"]
    atomic_write_csv(path, list(rows), fields)


def read_api_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _api_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("api_full_name", "") or row.get("api", "") or row.get("name", "") or row.get("full_name", "") or "").strip()


def read_api_set_from_csv(path: Path) -> Set[str]:
    return {_api_from_row(row) for row in read_csv_rows(path) if _api_from_row(row)}


def _pct(numerator: int, denominator: int) -> float:
    return round((float(numerator) / float(denominator)) * 100.0, 4) if denominator else 0.0


def write_text_list(path: Path, values: Sequence[str]) -> None:
    atomic_write_text(path, "".join(f"{value}\n" for value in values))


def dedupe_rows(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, ...]] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = tuple(str(row.get(field, "") or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _int(value: Any) -> int:
    try:
        return int(float(str(value or 0)))
    except Exception:
        return 0


def _candidate_bugs(bugs: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = bugs.get("candidate_bugs")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return [
        row
        for row in bugs.get("bugs", [])
        if isinstance(row, dict)
        and row.get("status") not in {"unresolved", "false_positive"}
        and row.get("category") not in {"unresolved_pipeline_failure", "invalid_generated_mutation", "expected_exception_on_invalid_input"}
    ]


def _stage3_seed_failures(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("stage3_status", "") or "") == "env_only" or str(row.get("smoke_classification", "") or "") == "environment_unsupported":
            continue
        out.append({
            "classification": "seed_generation_failure",
            "api_full_name": row.get("api_full_name", ""),
            "stage3_status": row.get("stage3_status", ""),
            "smoke_classification": row.get("smoke_classification", ""),
            "reason": row.get("reason", ""),
            "init_json_path": row.get("init_json_path", ""),
        })
    return out


def _stage3_platform_exclusions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("stage3_status", "") or "") != "env_only" and str(row.get("smoke_classification", "") or "") != "environment_unsupported":
            continue
        out.append({
            "classification": "environment_unsupported",
            "api_full_name": row.get("api_full_name", ""),
            "stage3_status": row.get("stage3_status", ""),
            "smoke_classification": row.get("smoke_classification", ""),
            "reason": row.get("reason", ""),
            "init_json_path": row.get("init_json_path", ""),
        })
    return out


def _stage3_scope_summary(
    ok_rows: Sequence[Dict[str, Any]],
    error_rows: Sequence[Dict[str, Any]],
    selected_apis: Set[str],
) -> Dict[str, Any]:
    def api(row: Dict[str, Any]) -> str:
        return str(row.get("api_full_name", "") or row.get("api", "") or "").strip()

    selected_ok = [row for row in ok_rows if api(row) in selected_apis] if selected_apis else list(ok_rows)
    selected_failed = [row for row in error_rows if api(row) in selected_apis] if selected_apis else list(error_rows)
    return {
        "global_ready": len(ok_rows),
        "global_failed": len(error_rows),
        "selected_ready": len(selected_ok),
        "selected_failed": len(selected_failed),
        "selected_apis": len(selected_apis) if selected_apis else len({api(row) for row in ok_rows if api(row)}),
    }


def _stage4_pipeline_errors(failure_rows: List[Dict[str, Any]], coverage: Dict[str, Any], bugs: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in failure_rows:
        stage = str(row.get("stage", "") or "")
        error_type = str(row.get("error_type", "") or "")
        if error_type == "invalid_valid_mutation" or stage == "mutation_generator":
            classification = "invalid_generated_input"
        elif stage == "materialize":
            classification = "seed_generation_failure"
        elif stage == "coverage":
            classification = "coverage_unavailable"
        else:
            classification = "pipeline_error"
        out.append({
            "classification": classification,
            "api_full_name": row.get("api_full_name", ""),
            "stage": stage,
            "mutation_intent": row.get("mutation_intent", ""),
            "param": row.get("param", ""),
            "rule": row.get("rule", ""),
            "error_type": error_type,
            "error": row.get("error", ""),
            "results_json_path": row.get("results_json_path", ""),
        })
    excluded = bugs.get("excluded_pipeline_issues", {})
    if isinstance(excluded, dict):
        for issue in excluded.get("issues", []) or []:
            if not isinstance(issue, dict):
                continue
            out.append({
                "classification": "pipeline_error",
                "api_full_name": issue.get("api", ""),
                "stage": issue.get("stage", ""),
                "error_type": issue.get("category", ""),
                "error": issue.get("stdout_stderr_excerpt", ""),
                "results_json_path": issue.get("testcase", ""),
            })
    if coverage.get("coverage_method") == "api_coverage_only" and coverage.get("line_coverage_percent") is None:
        out.append({
            "classification": "coverage_unavailable",
            "stage": "stage4_coverage",
            "error_type": "source_line_coverage_unavailable",
            "error": "Source-line coverage was not collected; API execution coverage is reported separately.",
        })
    return out


AUDIT_CLASSIFICATIONS = {
    "expected_negative_rejection",
    "negative_accepted_not_bug",
    "negative_mutation_accepted",
    "invalid_valid_mutation",
    "differential_mismatch",
}


def _stage4_events(
    failure_rows: Sequence[Dict[str, Any]],
    coverage: Dict[str, Any],
    bugs: Dict[str, Any],
    selected_apis: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pipeline: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    shortfalls: List[Dict[str, Any]] = []

    for row in failure_rows:
        api = str(row.get("api_full_name", "") or row.get("api", "") or "").strip()
        if selected_apis and api and api not in selected_apis:
            continue
        classification = str(row.get("classification", "") or row.get("error_type", "") or "").strip()
        item = dict(row)
        item.setdefault("classification", classification or "pipeline_error")
        if classification in AUDIT_CLASSIFICATIONS:
            audit.append(item)
            continue
        pipeline.extend(_stage4_pipeline_errors([item], coverage, {"excluded_pipeline_issues": {"issues": []}}))

    excluded = bugs.get("excluded_pipeline_issues", {})
    if isinstance(excluded, dict):
        for issue in excluded.get("issues", []) or []:
            if not isinstance(issue, dict):
                continue
            api = str(issue.get("api", "") or issue.get("api_full_name", "") or "").strip()
            if selected_apis and api and api not in selected_apis:
                continue
            message = str(issue.get("stdout_stderr_excerpt", "") or issue.get("error", "") or "")
            item = {
                "classification": "coverage_shortfall" if "selected_function_line_coverage_percent" in message else "pipeline_error",
                "api_full_name": api,
                "stage": issue.get("stage", ""),
                "error_type": issue.get("category", ""),
                "error": message,
                "results_json_path": issue.get("testcase", ""),
            }
            if item["classification"] == "coverage_shortfall":
                shortfalls.append(item)
            else:
                pipeline.append(item)

    if coverage.get("coverage_method") == "api_coverage_only" and coverage.get("line_coverage_percent") is None:
        pipeline.append({
            "classification": "coverage_unavailable",
            "stage": "stage4_coverage",
            "error_type": "source_line_coverage_unavailable",
            "error": "Source-line coverage was not collected; API execution coverage is reported separately.",
        })
    return pipeline, audit, shortfalls


def _coverage_payload(coverage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_execution_coverage_percent": coverage.get("api_execution_coverage_percent", coverage.get("api_coverage_percent")),
        "python_line_coverage_percent": coverage.get("python_line_coverage_percent"),
        "native_line_coverage_percent": coverage.get("native_line_coverage_percent"),
        "selected_function_line_coverage_percent": coverage.get("function_line_coverage_percent"),
        "coverage_method": coverage.get("coverage_method", "unavailable"),
        "coverage_available": coverage.get("coverage_available", {
            "api_execution": coverage.get("api_coverage_percent") is not None,
            "selected_function": coverage.get("function_line_coverage_percent") is not None,
            "python": coverage.get("python_line_coverage_percent") is not None,
            "native": coverage.get("native_line_coverage_percent") is not None,
        }),
        "limitations": coverage.get("coverage_limitations", coverage.get("limitations", [])),
        "warnings": coverage.get("coverage_warnings", []),
        "total_valid_programs": coverage.get("total_valid_programs", coverage.get("unique_valid_programs", 0)),
        "unique_valid_programs": coverage.get("unique_valid_programs", 0),
        "oracle_summary": coverage.get("oracle_summary", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate final DeepFuzz thesis summary")
    ap.add_argument("library", nargs="?", default=os.environ.get("LIB", "torch"))
    ap.add_argument("--lib", dest="library_opt", default="")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "latest"))
    ap.add_argument("--stage4-results-dir", default=os.environ.get("STAGE4_RESULTS_DIR", ""))
    args = ap.parse_args()

    lib = args.library_opt or args.library
    run_dir = ROOT / "pipeline_runs" / lib / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stage4_dir = Path(args.stage4_results_dir) if args.stage4_results_dir else ROOT / "stage4" / "results" / f"{lib}-thesis-all-coverage"
    if not stage4_dir.exists():
        stage4_dir = ROOT / "stage4" / "results" / f"{lib}-coverage"

    api_list_path = run_dir / "api_list.txt"
    selected_apis = read_api_list(api_list_path)
    accepted_csv = ROOT / "doc2info" / "results" / lib / "accepted.csv"
    accepted_apis = read_api_set_from_csv(accepted_csv)
    report_scope_apis = accepted_apis or selected_apis
    coverage = load_json(stage4_dir / "coverage_report.json")
    bugs = load_json(stage4_dir / "bug_report.json")
    stage2_ok = read_csv_rows(ROOT / "json_validator" / "results" / lib / "ok.csv")
    stage2_errors = read_csv_rows(ROOT / "json_validator" / "results" / lib / "errors.csv")
    stage3_errors_all = read_csv_rows(ROOT / "json2init" / "results" / lib / "errors.csv")
    stage3_ok = read_csv_rows(ROOT / "json2init" / "results" / lib / "ok.csv")
    stage3_errors = [
        row for row in stage3_errors_all
        if not report_scope_apis or _api_from_row(row) in report_scope_apis
    ]
    stage4_failures = read_csv_rows(stage4_dir / "failure.csv") + read_csv_rows(stage4_dir / "failures.csv")
    stage4_results = read_csv_rows(stage4_dir / "results.csv")
    stage2_ok_apis = {_api_from_row(row) for row in stage2_ok if _api_from_row(row)}
    stage2_error_apis = {_api_from_row(row) for row in stage2_errors if _api_from_row(row)}
    stage3_ok_apis = {_api_from_row(row) for row in stage3_ok if _api_from_row(row)}
    stage3_error_apis = {_api_from_row(row) for row in stage3_errors_all if _api_from_row(row)}
    stage4_result_apis = {_api_from_row(row) for row in stage4_results if _api_from_row(row)}
    accepted_not_stage3 = sorted((accepted_apis - stage3_ok_apis) if accepted_apis else stage3_error_apis)
    selected_stage3_ready_apis = (stage3_ok_apis & selected_apis) if selected_apis else set(stage3_ok_apis)
    stage3_ready_outside_frozen = sorted((stage3_ok_apis - selected_apis) if selected_apis else [])
    stage3_ready_not_stage4 = sorted(selected_stage3_ready_apis - stage4_result_apis)
    internal_excluded_before_stage4 = [api for api in stage3_ready_not_stage4 if is_internal_api(api, lib)]
    stage4_not_executed_noninternal = [api for api in stage3_ready_not_stage4 if not is_internal_api(api, lib)]
    accepted_denominator = len(accepted_apis) or len(report_scope_apis) or len(stage4_result_apis)
    stage4_summary = load_json(stage4_dir / "summary.json")
    stage3_summary = csv_stage_summary(
        "stage3_init_generation",
        ROOT / "json2init" / "results" / lib / "ok.csv",
        ROOT / "json2init" / "results" / lib / "errors.csv",
    )
    stage3_scope = _stage3_scope_summary(stage3_ok, stage3_errors_all, selected_apis)
    candidate_bugs = _candidate_bugs(bugs)
    stage4_pipeline, audit_events, coverage_shortfalls = _stage4_events(stage4_failures, coverage, bugs, stage4_result_apis or selected_apis)
    invalid_valid_mutations = dedupe_rows([
        row for row in audit_events + stage4_pipeline
        if str(row.get("classification", "") or row.get("error_type", "")) in {"invalid_valid_mutation", "invalid_generated_input"}
    ], ("api_full_name", "mutation_intent", "param", "rule"))
    negative_accepted = [
        row for row in audit_events
        if str(row.get("classification", "") or row.get("error_type", "")) in {"negative_accepted_not_bug", "negative_mutation_accepted"}
    ]
    pipeline_errors = _stage3_seed_failures(stage3_errors) + [
        row for row in stage4_pipeline
        if str(row.get("classification", "") or "") not in {"invalid_generated_input"}
    ] + coverage_shortfalls
    stage3_platform_exclusions = _stage3_platform_exclusions(stage3_errors)
    interesting_seeds = [
        {
            "api_full_name": row.get("api_full_name", ""),
            "valid_programs": _int(row.get("valid_programs")),
            "results_json_path": row.get("results_json_path", ""),
        }
        for row in stage4_results
        if _int(row.get("valid_programs")) > 0
    ][:50]
    triage_dir = run_dir / "triage"
    coverage_dir = run_dir / "coverage"
    generated_tests_dir = run_dir / "generated_tests"
    repro_dir = run_dir / "repro"
    for path in (triage_dir, coverage_dir, generated_tests_dir, repro_dir, coverage_dir / "html"):
        path.mkdir(parents=True, exist_ok=True)

    candidate_doc_mismatches = [
        row for row in candidate_bugs
        if str(row.get("category", "") or "") in {"candidate_documentation_mismatch", "documentation_mismatch"}
        or str(row.get("oracle_verdict", "") or "") == "candidate_documentation_mismatch"
    ]
    candidate_impl_bugs = [
        row for row in candidate_bugs
        if row not in candidate_doc_mismatches
    ]
    write_csv_rows(triage_dir / "candidate_bugs.csv", candidate_impl_bugs)
    write_csv_rows(triage_dir / "candidate_doc_mismatches.csv", candidate_doc_mismatches)
    write_csv_rows(triage_dir / "pipeline_failures.csv", pipeline_errors)
    write_csv_rows(triage_dir / "stage3_seed_failures.csv", _stage3_seed_failures(stage3_errors))
    write_csv_rows(triage_dir / "stage3_platform_exclusions.csv", stage3_platform_exclusions)
    write_csv_rows(triage_dir / "invalid_valid_mutations.csv", invalid_valid_mutations)
    write_csv_rows(triage_dir / "negative_accepted_not_bug.csv", negative_accepted)
    write_text_list(triage_dir / "accepted_not_stage3.txt", accepted_not_stage3)
    write_text_list(triage_dir / "internal_excluded_before_stage4.txt", internal_excluded_before_stage4)
    write_text_list(triage_dir / "stage4_not_executed_noninternal.txt", stage4_not_executed_noninternal)
    write_text_list(triage_dir / "stage3_ready_outside_frozen.txt", stage3_ready_outside_frozen)
    selected_function_payload = coverage.get("selected_function_coverage", {})
    if not selected_function_payload:
        selected_function_payload = {
            "available": False,
            "reason": "selected_function_coverage_not_present",
            "excluded_unavailable_apis": coverage.get("coverage_summary", {}).get("selected_function_line_coverage", {}).get("unavailable_apis", []),
        }
    write_json(coverage_dir / "selected_function_coverage.json", selected_function_payload)

    generated_test_cases = len(list(generated_tests_dir.rglob("test_*.py"))) if generated_tests_dir.exists() else 0
    ready_seeds = stage3_scope.get("selected_ready", stage3_summary["total_valid"])
    base_valid_apis = _int(stage4_summary.get("base_success"))
    api_list_entries = len(selected_apis) if selected_apis else _int(stage4_summary.get("selected_apis"))
    stage4_evaluated_apis = _int(stage4_summary.get("selected_apis")) or len(stage4_result_apis)
    frozen_apis = api_list_entries or accepted_denominator or stage4_evaluated_apis
    evaluated_api_execution = coverage.get("api_execution_coverage_percent", coverage.get("api_coverage_percent"))
    final_table = {
        "Library": lib,
        "Frozen APIs": frozen_apis,
        "Accepted documented APIs": accepted_denominator,
        "Stage 2 valid specs": f"{len(stage2_ok_apis)}/{accepted_denominator} ({_pct(len(stage2_ok_apis), accepted_denominator)}%)",
        "Stage 2 failures": len(stage2_error_apis),
        "Input API list entries": api_list_entries,
        "Ready seeds": f"{len(stage3_ok_apis)}/{accepted_denominator} ({_pct(len(stage3_ok_apis), accepted_denominator)}%)",
        "Frozen entries ready for Stage 4": f"{len(selected_stage3_ready_apis)}/{frozen_apis} ({_pct(len(selected_stage3_ready_apis), frozen_apis)}%)",
        "Stage 3 non-ready APIs": len(accepted_not_stage3),
        "Platform/runtime exclusions": len(stage3_platform_exclusions),
        "Stage 3 ready APIs outside frozen list": len(stage3_ready_outside_frozen),
        "Stage 4 evaluated APIs": (
            f"{stage4_evaluated_apis}/{accepted_denominator} ({_pct(stage4_evaluated_apis, accepted_denominator)}%) accepted; "
            f"{stage4_evaluated_apis}/{frozen_apis} ({_pct(stage4_evaluated_apis, frozen_apis)}%) frozen"
        ),
        "Internal frozen APIs excluded before Stage 4": len(internal_excluded_before_stage4),
        "Non-internal frozen ready APIs not executed": len(stage4_not_executed_noninternal),
        "Base-valid APIs": (
            f"{base_valid_apis}/{accepted_denominator} ({_pct(base_valid_apis, accepted_denominator)}%) accepted; "
            f"{base_valid_apis}/{stage4_evaluated_apis} ({_pct(base_valid_apis, stage4_evaluated_apis)}%) evaluated"
        ),
        "Accepted API execution coverage": _pct(base_valid_apis, accepted_denominator),
        "Frozen API execution coverage": _pct(base_valid_apis, frozen_apis),
        "Evaluated API execution coverage": evaluated_api_execution,
        "Generated test cases": generated_test_cases,
        "Valid programs": _int(stage4_summary.get("total_valid_programs", stage4_summary.get("valid_programs"))),
        "Expected negative rejections": _int(stage4_summary.get("expected_negative_rejection")),
        "Invalid valid mutations": _int(stage4_summary.get("invalid_valid_mutation")),
        "Device oracle mismatches": _int(stage4_summary.get("device_oracle_mismatch")),
        "Negative accepted not bug": _int(stage4_summary.get("negative_accepted_not_bug")),
        "Candidate doc mismatches": len(candidate_doc_mismatches),
        "Candidate implementation bugs": len(candidate_impl_bugs),
        "API execution coverage": evaluated_api_execution,
        "Selected-function line coverage": coverage.get("function_line_coverage_percent"),
        "Package Python coverage": coverage.get("python_line_coverage_percent"),
        "Runtime": coverage.get("execution_time_seconds", 0.0),
    }
    payload = {
        "schema_version": "next",
        "library": lib,
        "run": {
            "run_id": args.run_id,
            "stage4_results_dir": str(stage4_dir),
            "seed": stage4_summary.get("seed", ""),
            "mutation_budget": stage4_summary.get("mutation_budget", ""),
        },
        "pipeline_health": {
            "accepted": {
                "accepted_csv": str(accepted_csv),
                "accepted_apis": accepted_denominator,
            },
            "stage1": load_json(ROOT / "info2json" / "results" / lib / "summary.json"),
            "stage2": csv_stage_summary(
                "stage2_json_validation",
                ROOT / "json_validator" / "results" / lib / "ok.csv",
                ROOT / "json_validator" / "results" / lib / "errors.csv",
            ),
            "stage3": {
                "valid": stage3_summary["total_valid"],
                "failed": stage3_summary["total_failed"],
                "total_seen": stage3_summary["total_seen"],
                "scope": stage3_scope,
                "accepted_not_stage3_count": len(accepted_not_stage3),
                "seed_generation_failures": _stage3_seed_failures(stage3_errors),
                "platform_exclusions": stage3_platform_exclusions,
            },
            "stage4": {
                "executed_apis": stage4_summary.get("executed_apis", 0),
                "accepted_denominator": accepted_denominator,
                "frozen_api_list_entries": frozen_apis,
                "accepted_to_stage4_percent": _pct(stage4_evaluated_apis, accepted_denominator),
                "frozen_to_stage4_percent": _pct(stage4_evaluated_apis, frozen_apis),
                "stage3_ready_outside_frozen": len(stage3_ready_outside_frozen),
                "internal_excluded_before_stage4": len(internal_excluded_before_stage4),
                "noninternal_ready_not_executed": len(stage4_not_executed_noninternal),
                "base_success": stage4_summary.get("base_success", 0),
                "mutation_cases": stage4_summary.get("mutation_cases", 0),
                "valid_mutations": stage4_summary.get("mutation_success", 0),
                "invalid_generated_inputs": stage4_summary.get("invalid_valid_mutation", 0),
                "device_oracle_mismatches": stage4_summary.get("device_oracle_mismatch", 0),
                "expected_negative_rejections": stage4_summary.get("expected_negative_rejection", 0),
                "negative_accepted_not_bug": stage4_summary.get("negative_accepted_not_bug", 0),
                "real_bug_candidates": len(candidate_bugs),
                "total_valid_programs": stage4_summary.get("total_valid_programs", stage4_summary.get("valid_programs", 0)),
                "unique_valid_programs": stage4_summary.get("unique_valid_programs", 0),
            },
        },
        "final_evaluation_table": final_table,
        "coverage": _coverage_payload(coverage),
        "candidate_bugs": candidate_bugs,
        "candidate_documentation_mismatches": candidate_doc_mismatches,
        "pipeline_errors": pipeline_errors,
        "platform_exclusions": stage3_platform_exclusions,
        "invalid_valid_mutations": invalid_valid_mutations,
        "negative_accepted_not_bug": negative_accepted,
        "coverage_shortfalls": coverage_shortfalls,
        "interesting_seeds": interesting_seeds,
        "repair_summary": load_json(run_dir / "repair" / "repair_summary.json"),
        "run_manifest": load_json(run_dir / "run_manifest.json"),
        "artifact_paths": {
            "api_list": str(api_list_path),
            "run_manifest": str(run_dir / "run_manifest.json"),
            "coverage_report": str(stage4_dir / "coverage_report.json"),
            "coverage_markdown": str(stage4_dir / "coverage_report.md"),
            "selected_function_coverage": str(coverage_dir / "selected_function_coverage.json"),
            "bug_report": str(stage4_dir / "bug_report.json"),
            "bug_report_csv": str(stage4_dir / "bug_report.csv"),
            "bug_audit_csv": str(stage4_dir / "bug_audit.csv"),
            "results_csv": str(stage4_dir / "results.csv"),
            "failures_csv": str(stage4_dir / "failures.csv"),
            "failure_audit_csv": str(stage4_dir / "failure.csv"),
            "stage3_seed_failures_csv": str(triage_dir / "stage3_seed_failures.csv"),
            "stage3_platform_exclusions_csv": str(triage_dir / "stage3_platform_exclusions.csv"),
            "accepted_not_stage3": str(triage_dir / "accepted_not_stage3.txt"),
            "internal_excluded_before_stage4": str(triage_dir / "internal_excluded_before_stage4.txt"),
            "stage4_not_executed_noninternal": str(triage_dir / "stage4_not_executed_noninternal.txt"),
            "stage3_ready_outside_frozen": str(triage_dir / "stage3_ready_outside_frozen.txt"),
            "generated_tests": str(generated_tests_dir),
            "triage": str(triage_dir),
            "final_report": str(run_dir / "final_report.json"),
        },
    }
    out_json = run_dir / "final_report.json"
    write_json(out_json, payload)

    coverage_method = coverage.get("coverage_method", "unavailable")
    lines = [
        "# DeepFuzz Final Report",
        "",
        "DeepFuzz reports selected-function Python line coverage over the APIs that reached Stage 4. Accepted documented APIs, frozen run-list entries, and evaluated/base-valid APIs are reported separately so the funnel remains auditable.",
        "",
        f"- Library: {lib}",
        f"- Frozen APIs: {final_table['Frozen APIs']}",
        f"- Accepted documented APIs: {final_table['Accepted documented APIs']}",
        f"- Stage 2 valid specs: {final_table['Stage 2 valid specs']}",
        f"- Stage 3 ready seeds: {final_table['Ready seeds']}",
        f"- Stage 4 evaluated APIs: {final_table['Stage 4 evaluated APIs']}",
        f"- Coverage metric: {coverage.get('main_coverage_number', 0.0)} ({coverage.get('main_coverage_label', 'unavailable')})",
        f"- Coverage method: {coverage_method}",
        f"- Base-valid APIs: {final_table['Base-valid APIs']}",
        f"- API execution coverage over evaluated APIs: {coverage.get('covered_api_count', 0)}/{coverage.get('total_api_count', 0)} ({evaluated_api_execution}%)",
        f"- Selected-function line coverage: {coverage.get('function_line_coverage_percent', 'unavailable')}",
        f"- Package Python coverage: {coverage.get('python_line_coverage_percent', 'unavailable')}",
        f"- Total valid programs: {payload['coverage'].get('total_valid_programs', 0)}",
        f"- Unique valid programs: {payload['coverage'].get('unique_valid_programs', 0)}",
        f"- Expected negative rejections: {stage4_summary.get('expected_negative_rejection', 0)}",
        f"- Invalid valid mutations: {stage4_summary.get('invalid_valid_mutation', 0)}",
        f"- Device oracle mismatches: {stage4_summary.get('device_oracle_mismatch', 0)}",
        f"- Negative accepted not bug: {stage4_summary.get('negative_accepted_not_bug', 0)}",
        f"- Candidate implementation bugs: {len(candidate_impl_bugs)}",
        f"- Candidate documentation mismatches: {len(candidate_doc_mismatches)}",
        f"- Pipeline errors: {len(pipeline_errors)}",
        f"- Platform/runtime exclusions: {len(stage3_platform_exclusions)}",
        f"- Execution time: {coverage.get('execution_time_seconds', 0.0)} seconds",
        "",
        "## API Funnel",
        "",
        "| Stage | APIs | Notes |",
        "| --- | ---: | --- |",
        f"| Accepted documented APIs | {accepted_denominator} | Baseline denominator from `{accepted_csv}` |",
        f"| Frozen API list entries | {frozen_apis} | Entries in `{api_list_path}` |",
        f"| Stage 2 valid specs | {len(stage2_ok_apis)} | {len(stage2_error_apis)} Stage 2 failures |",
        f"| Stage 3 ready seeds | {len(stage3_ok_apis)} | {len(_stage3_seed_failures(stage3_errors))} seed/init failures; {len(stage3_platform_exclusions)} platform/runtime exclusions |",
        f"| Frozen entries ready for Stage 4 | {len(selected_stage3_ready_apis)} | {len(stage3_ready_outside_frozen)} Stage 3-ready APIs are outside the frozen list |",
        f"| Stage 4 evaluated APIs | {stage4_evaluated_apis} | {len(internal_excluded_before_stage4)} frozen internal APIs excluded before Stage 4 |",
        f"| Stage 4 base-valid APIs | {base_valid_apis} | {len(stage4_not_executed_noninternal)} non-internal frozen ready APIs not executed |",
        "",
        "## Final Evaluation Table",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines.extend([f"| {key} | {value} |" for key, value in final_table.items()])
    lines.extend([
        "",
        "## Artifacts",
        f"- Coverage JSON: {stage4_dir / 'coverage_report.json'}",
        f"- Selected-function coverage JSON: {coverage_dir / 'selected_function_coverage.json'}",
        f"- Bug JSON: {stage4_dir / 'bug_report.json'}",
        f"- Bug CSV: {stage4_dir / 'bug_report.csv'}",
        f"- Triage: {triage_dir}",
        f"- Stage 3 seed failures: {triage_dir / 'stage3_seed_failures.csv'}",
        f"- Stage 3 platform exclusions: {triage_dir / 'stage3_platform_exclusions.csv'}",
        f"- Accepted APIs missing Stage 3: {triage_dir / 'accepted_not_stage3.txt'}",
        f"- Internal APIs excluded before Stage 4: {triage_dir / 'internal_excluded_before_stage4.txt'}",
        f"- Stage 3-ready APIs outside frozen list: {triage_dir / 'stage3_ready_outside_frozen.txt'}",
        f"- Generated tests: {generated_tests_dir}",
        f"- Final JSON: {out_json}",
    ])
    if coverage_method == "api_coverage_only":
        lines.extend(["", "## Coverage Interpretation"])
        lines.append(f"- {coverage.get('api_coverage_percent', 0.0)}% API execution coverage over the accepted API subset; this is not source-line/code coverage.")
    limitations = coverage.get("coverage_limitations") if isinstance(coverage, dict) else []
    if limitations:
        lines.extend(["", "## Coverage Limitations"])
        lines.extend([f"- {item}" for item in limitations])
    out_md = run_dir / "final_report.md"
    atomic_write_text(out_md, "\n".join(lines) + "\n")
    print(f"[report] wrote {out_json}")
    print(f"[report] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
