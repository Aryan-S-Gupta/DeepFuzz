from __future__ import annotations

import csv
import inspect
import importlib
import json
import sys
from pathlib import Path

import pytest

from json2init.deepfuzz_common import build_runtime_object_from_spec, instantiate_base_call
from json2init.json2init import process_specs
from scripts.report import _stage3_scope_summary, _stage4_events
from scripts.repair import _coverage_report_errors, is_expected_negative_stage4
from stage4.stage4_coverage_runner import build_coverage_report, run_stage4


FAKE_MODULE = "tests.fixtures.fake_stage4_api"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_spec(spec_dir: Path, api: str, params: dict[str, dict[str, object]] | None = None) -> Path:
    payload = {
        "api_name": api.rsplit(".", 1)[-1],
        "module_path": api.rsplit(".", 1)[0],
        "params": params or {"x": {"type": "int", "flag": "Required", "description": "integer input", "constraints": []}},
        "output": {"type": "int", "numbers": "1", "description": "result"},
        "constraints": [],
    }
    path = spec_dir / f"{api}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_init(path: Path, api: str, *, ready: bool = True, valid_rules: list[str] | None = None, negative_rules: list[str] | None = None) -> None:
    payload = {
        "api_full_name": api,
        "api_name": api.rsplit(".", 1)[-1],
        "module_path": api.rsplit(".", 1)[0],
        "import_path": api,
        "backend": "python",
        "params": {
            "x": {
                "name": "x",
                "spec": {"type": "int", "flag": "Required", "description": "integer input"},
                "chosen_type": "int",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "literal", "value": 1},
                "mutation_rules": (valid_rules or ["scalar_delta"]) + (negative_rules or ["type_widen"]),
                "valid_mutation_rules": valid_rules if valid_rules is not None else ["scalar_delta"],
                "negative_mutation_rules": negative_rules if negative_rules is not None else ["type_widen"],
                "unresolved_reason": "",
                "call_kind": "keyword",
            }
        },
        "output": {"type": "int"},
        "constraints": [],
        "spec_ready": ready,
        "ready_for_stage4": ready,
        "readiness_reasons": [] if ready else ["forced not ready"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage3_deterministic_seed_generation_and_positional_materialization():
    api = f"{FAKE_MODULE}.positional_only_api"
    spec = {
        "api_name": "positional_only_api",
        "module_path": FAKE_MODULE,
        "params": {"x": {"type": "int", "flag": "Required", "description": "integer input", "constraints": []}},
        "output": {"type": "int", "description": ""},
        "constraints": [],
    }

    first = build_runtime_object_from_spec(api, spec)
    second = build_runtime_object_from_spec(api, spec)
    args, kwargs, reasons = instantiate_base_call(first)

    assert first.to_dict() == second.to_dict()
    assert reasons == []
    assert args == [1]
    assert kwargs == {}
    assert first.params["x"].call_kind == "positional_only"


def test_stage3_smoke_test_gates_bad_seed(tmp_path):
    spec_dir = tmp_path / "specs"
    outdir = tmp_path / "init"
    api = f"{FAKE_MODULE}.requires_even"
    _write_spec(spec_dir, api)
    ok_csv = tmp_path / "ok.csv"
    _write_csv(ok_csv, ["api_full_name"], [{"api_full_name": api}])

    process_specs(str(spec_dir), str(outdir), smoke_test=True, ok_csv=str(ok_csv), smoke_timeout_sec=10)

    assert _read_csv(outdir / "ok.csv") == []
    errors = _read_csv(outdir / "errors.csv")
    assert errors[0]["api_full_name"] == api
    assert errors[0]["stage3_status"] == "api_runtime_error"
    init = json.loads((outdir / f"{api}.init.json").read_text(encoding="utf-8"))
    assert init["ready_for_stage4"] is False


def test_stage3_reuse_existing_force_and_limit(tmp_path):
    spec_dir = tmp_path / "specs"
    outdir = tmp_path / "init"
    apis = [f"{FAKE_MODULE}.add_one", f"{FAKE_MODULE}.int_only"]
    for api in apis:
        _write_spec(spec_dir, api)
    ok_csv = tmp_path / "ok.csv"
    _write_csv(ok_csv, ["api_full_name"], [{"api_full_name": api} for api in apis])
    stale = outdir / f"{apis[0]}.init.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"api_full_name": apis[0], "ready_for_stage4": True, "sentinel": "old"}), encoding="utf-8")

    process_specs(str(spec_dir), str(outdir), ok_csv=str(ok_csv), smoke_test=False, reuse_existing=True, limit=1)
    assert json.loads(stale.read_text(encoding="utf-8"))["sentinel"] == "old"
    assert len(_read_csv(outdir / "ok.csv")) == 1

    process_specs(str(spec_dir), str(outdir), ok_csv=str(ok_csv), smoke_test=False, overwrite=True, limit=1)
    assert "sentinel" not in json.loads(stale.read_text(encoding="utf-8"))


def test_stage4_execution_summary_failure_aliases_and_negative_rejection(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    _write_init(init_dir / f"{FAKE_MODULE}.int_only.init.json", f"{FAKE_MODULE}.int_only")

    run_stage4(str(init_dir), str(results_dir), mutation_budget=1, seed=1337, coverage_scope="api_only", worker_timeout_sec=20)

    assert (results_dir / "execution_summary.csv").exists()
    assert (results_dir / "failure.csv").exists()
    assert (results_dir / "results.csv").exists()
    assert (results_dir / "failures.csv").exists()
    execution = _read_csv(results_dir / "execution_summary.csv")
    failure = _read_csv(results_dir / "failure.csv")
    assert {row["case_kind"] for row in execution} >= {"base_valid", "valid_mutation", "negative_mutation"}
    assert any(row["classification"] == "expected_negative_rejection" for row in failure)
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))
    assert bugs["summary"]["total_candidates"] == 0
    assert _read_csv(results_dir / "bug_report.csv") == []


def test_stage4_candidate_bug_has_repro_and_is_not_failure_row(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    _write_init(init_dir / f"{FAKE_MODULE}.abort_on_two.init.json", f"{FAKE_MODULE}.abort_on_two", negative_rules=[])

    run_stage4(str(init_dir), str(results_dir), mutation_budget=1, seed=1337, coverage_scope="api_only", worker_timeout_sec=20, case_timeout_sec=5)

    bug_rows = _read_csv(results_dir / "bug_report.csv")
    assert len(bug_rows) == 1
    assert bug_rows[0]["category"] == "crash_or_abort"
    assert Path(bug_rows[0]["reproducer_path"]).exists()
    failure_text = (results_dir / "failure.csv").read_text(encoding="utf-8")
    assert "crash_or_abort" not in failure_text


def test_stage4_pipeline_failure_excluded_from_bug_report(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    _write_init(init_dir / f"{FAKE_MODULE}.add_one.init.json", f"{FAKE_MODULE}.add_one", ready=False)

    run_stage4(str(init_dir), str(results_dir), mutation_budget=1, seed=1337, coverage_scope="api_only", worker_timeout_sec=20)

    assert _read_csv(results_dir / "bug_report.csv") == []
    failure = _read_csv(results_dir / "failure.csv")
    assert failure[0]["classification"] == "seed_materialization_failure"


def test_final_report_helpers_scope_stage3_and_audit_negative_acceptance():
    selected = {f"{FAKE_MODULE}.add_one"}
    stage3 = _stage3_scope_summary(
        ok_rows=[{"api_full_name": f"{FAKE_MODULE}.add_one"}],
        error_rows=[{"api_full_name": f"{FAKE_MODULE}.int_only", "reason": "not selected"}],
        selected_apis=selected,
    )
    pipeline, audit, shortfalls = _stage4_events(
        failure_rows=[
            {
                "api_full_name": f"{FAKE_MODULE}.add_one",
                "stage": "negative_mutation",
                "classification": "negative_mutation_accepted",
                "error_type": "negative_mutation_accepted",
                "error": "negative mutation executed without rejection",
            },
            {
                "api_full_name": f"{FAKE_MODULE}.int_only",
                "stage": "materialize",
                "classification": "seed_materialization_failure",
                "error_type": "materialization_error",
                "error": "not selected",
            },
        ],
        coverage={"coverage_method": "python_coverage"},
        bugs={
            "excluded_pipeline_issues": {
                "issues": [
                    {
                        "api": f"{FAKE_MODULE}.add_one",
                        "stage": "stage4_coverage",
                        "category": "unresolved_pipeline_failure",
                        "stdout_stderr_excerpt": "selected_function_line_coverage_percent=25 below threshold 90",
                    }
                ]
            }
        },
        selected_apis=selected,
    )

    assert stage3["selected_failed"] == 0
    assert stage3["global_failed"] == 1
    assert pipeline == []
    assert audit[0]["classification"] == "negative_mutation_accepted"
    assert shortfalls[0]["classification"] == "coverage_shortfall"


def test_repair_ignores_negative_acceptance_and_targets_function_coverage(tmp_path, monkeypatch):
    assert is_expected_negative_stage4(
        {"mutation_intent": "negative", "error_type": "negative_mutation_accepted"},
        "stage4_runtime",
    )
    monkeypatch.setenv("DEEPFUZZ_FUNCTION_COVERAGE_THRESHOLD", "90")
    results_dir = tmp_path / "coverage"
    results_dir.mkdir()
    (results_dir / "coverage_report.json").write_text(
        json.dumps({
            "per_api": [
                {
                    "api": f"{FAKE_MODULE}.add_one",
                    "covered": True,
                    "function_line_coverage_percent": 100.0,
                    "function_missing_lines": [],
                },
                {
                    "api": f"{FAKE_MODULE}.int_only",
                    "covered": True,
                    "function_line_coverage_percent": 25.0,
                    "function_missing_lines": [10, 11],
                },
            ]
        }),
        encoding="utf-8",
    )

    rows = _coverage_report_errors("fake", str(results_dir))

    assert len(rows) == 1
    assert rows[0]["api_full_name"] == f"{FAKE_MODULE}.int_only"
    assert rows[0]["stage"] == "stage4_coverage"
    assert "selected_function_line_coverage_percent=25" in rows[0]["message"]


def test_stage4_api_only_coverage_labels_and_null_line_coverage(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    _write_init(init_dir / f"{FAKE_MODULE}.add_one.init.json", f"{FAKE_MODULE}.add_one", valid_rules=[], negative_rules=[])

    run_stage4(str(init_dir), str(results_dir), mutation_budget=0, seed=1337, coverage_scope="api_only", worker_timeout_sec=20)

    coverage = json.loads((results_dir / "coverage_report.json").read_text(encoding="utf-8"))
    markdown = (results_dir / "coverage_report.md").read_text(encoding="utf-8")
    assert coverage["coverage_method"] == "api_coverage_only"
    assert coverage["main_coverage_label"] == "API execution coverage %"
    assert coverage["line_coverage_percent"] is None
    assert coverage["python_line_coverage_percent"] is None
    assert "Python line coverage: unavailable" in markdown


def test_python_coverage_report_uses_line_coverage_as_source_metric():
    summary = {
        "selected_apis": 10,
        "base_success": 10,
        "total_valid_programs": 33,
        "unique_valid_programs": 33,
        "mutation_cases": 40,
        "expected_negative_rejection": 7,
        "invalid_mutation": 1,
        "seed": 1337,
        "mutation_budget": 4,
        "coverage_scope": "both",
    }
    coverage_rows = [
        {
            "api_full_name": "__CAMPAIGN__",
            "python_statement_percent": "0.0121",
            "python_covered_lines": "14",
            "python_coverable_lines": "115252",
        }
    ]
    results_rows = [
        {"api_full_name": f"jax.numpy.fake{i}", "base_success": "True", "valid_programs": "1", "backend": "jax"}
        for i in range(10)
    ]
    report = build_coverage_report(
        summary,
        coverage_rows,
        results_rows,
        {"summary": {"total_candidates": 0}, "candidate_bugs": []},
        execution_time_seconds=1.0,
        native_requested=True,
        python_requested=True,
        run_id="coverage-demo",
    )

    assert report["coverage_method"] == "python_coverage"
    assert report["main_coverage_label"] == "Python wrapper line coverage %"
    assert report["main_coverage_number"] == 0.0121
    assert report["api_execution_coverage_percent"] == 100.0
    assert report["coverage_summary"]["api_execution"]["percent"] == 100.0
    assert report["coverage_summary"]["python_line_coverage"]["percent"] == 0.0121
    assert report["coverage_summary"]["native_line_coverage"]["available"] is False


def test_python_coverage_report_prefers_selected_function_line_coverage(tmp_path):
    fake = importlib.import_module(FAKE_MODULE)
    source_file = inspect.getsourcefile(fake.add_one)
    source_lines, start_line = inspect.getsourcelines(fake.add_one)
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({
            "files": {
                source_file: {
                    "executed_lines": list(range(start_line, start_line + len(source_lines))),
                }
            },
            "totals": {"covered_percent": 1.0, "covered_lines": 1, "num_statements": 100},
        }),
        encoding="utf-8",
    )
    summary = {
        "selected_apis": 1,
        "base_success": 1,
        "total_valid_programs": 1,
        "unique_valid_programs": 1,
        "seed": 1337,
        "mutation_budget": 0,
        "coverage_scope": "python",
    }
    coverage_rows = [
        {
            "api_full_name": "__CAMPAIGN__",
            "python_statement_percent": "1.0",
            "python_covered_lines": "1",
            "python_coverable_lines": "100",
            "python_json_path": str(coverage_json),
        }
    ]
    results_rows = [
        {"api_full_name": f"{FAKE_MODULE}.add_one", "base_success": "True", "valid_programs": "1", "backend": "python"}
    ]

    report = build_coverage_report(
        summary,
        coverage_rows,
        results_rows,
        {"summary": {"total_candidates": 0}, "candidate_bugs": []},
        execution_time_seconds=1.0,
        native_requested=False,
        python_requested=True,
        run_id="function-coverage-demo",
    )

    assert report["main_coverage_label"] == "Selected function line coverage %"
    assert report["function_line_coverage_percent"] == 100.0
    assert report["coverage_available"]["selected_function"] is True
    assert report["coverage_summary"]["selected_function_line_coverage"]["available"] is True
    assert report["per_api"][0]["function_line_coverage_percent"] == 100.0


def test_stage4_small_jax_integration_when_installed(tmp_path):
    pytest.importorskip("jax")
    spec_dir = tmp_path / "specs"
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    api = "jax.numpy.negative"
    _write_spec(
        spec_dir,
        api,
        {"x": {"type": "tensor", "flag": "Required", "description": "numeric array", "constraints": []}},
    )
    ok_csv = tmp_path / "ok.csv"
    _write_csv(ok_csv, ["api_full_name"], [{"api_full_name": api}])

    process_specs(str(spec_dir), str(init_dir), smoke_test=True, ok_csv=str(ok_csv), smoke_timeout_sec=20)
    run_stage4(str(init_dir), str(results_dir), mutation_budget=1, seed=1337, ok_csv=str(init_dir / "ok.csv"), coverage_scope="api_only", worker_timeout_sec=30)

    coverage = json.loads((results_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert coverage["covered_api_count"] == 1
