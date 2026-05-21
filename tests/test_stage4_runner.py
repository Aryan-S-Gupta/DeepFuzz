from __future__ import annotations

import json
import csv
import os
import random
import subprocess
import sys
import pytest

from stage4.stage4_coverage_runner import build_bug_report, collect_environment_metadata, reclassify_bug_report, target_version_from_rows
from stage4.stage4_worker import choose_rules
from json2init.deepfuzz_common import mutate_value
from scripts.report import _coverage_payload, _stage4_pipeline_errors


def _write_init(path, api_name: str, import_path: str) -> None:
    payload = {
        "api_full_name": api_name,
        "api_name": import_path.rsplit(".", 1)[-1],
        "module_path": import_path.rsplit(".", 1)[0],
        "import_path": import_path,
        "backend": "python",
        "params": {
            "x": {
                "name": "x",
                "spec": {"type": "int", "flag": "Required", "description": "integer input"},
                "chosen_type": "int",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "literal", "value": 1},
                "mutation_rules": ["scalar_delta", "type_widen"],
                "valid_mutation_rules": ["scalar_delta"],
                "negative_mutation_rules": ["type_widen"],
                "unresolved_reason": "",
            }
        },
        "output": {"type": "int"},
        "constraints": [],
        "spec_ready": True,
        "ready_for_stage4": True,
        "readiness_reasons": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_padding_init(path) -> None:
    payload = {
        "api_full_name": "jax.lax.padtype_to_pads",
        "api_name": "padding_api",
        "module_path": "tests.fixtures.fake_stage4_api",
        "import_path": "tests.fixtures.fake_stage4_api.padding_api",
        "backend": "jax",
        "params": {
            "padding": {
                "name": "padding",
                "spec": {
                    "type": "string",
                    "flag": "Required",
                    "description": "padding mode",
                    "enum_values": ["VALID", "SAME", "SAME_LOWER"],
                    "case_insensitive": True,
                    "constraints": ["padding must be one of VALID, SAME, SAME_LOWER"],
                },
                "chosen_type": "string",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "literal", "value": "VALID"},
                "mutation_rules": ["enum_switch", "enum_invalid"],
                "valid_mutation_rules": ["enum_switch"],
                "negative_mutation_rules": ["enum_invalid"],
                "unresolved_reason": "",
            }
        },
        "output": {"type": "string"},
        "constraints": [],
        "spec_ready": True,
        "ready_for_stage4": True,
        "readiness_reasons": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_payload(path, api_name: str, import_path: str, params: dict) -> None:
    path.write_text(json.dumps({
        "api_full_name": api_name,
        "api_name": import_path.rsplit(".", 1)[-1],
        "module_path": import_path.rsplit(".", 1)[0],
        "import_path": import_path,
        "backend": "python",
        "params": params,
        "output": {"type": "any"},
        "constraints": [],
        "spec_ready": True,
        "ready_for_stage4": True,
        "readiness_reasons": [],
    }), encoding="utf-8")


def test_stage4_no_coverage_writes_artifacts(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(init_dir / "tests.fixtures.fake_stage4_api.add_one.init.json", "tests.fixtures.fake_stage4_api.add_one", "tests.fixtures.fake_stage4_api.add_one")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--mutation-budget",
            "1",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    assert (results_dir / "results.csv").exists()
    assert (results_dir / "failures.csv").exists()
    coverage = json.loads((results_dir / "coverage_report.json").read_text(encoding="utf-8"))
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))
    assert coverage["schema_version"] == "2.0"
    assert coverage["coverage_method"] == "api_coverage_only"
    assert coverage["api_execution_coverage_percent"] == coverage["api_coverage_percent"]
    assert coverage["python_line_coverage_percent"] is None
    assert coverage["native_line_coverage_percent"] is None
    assert coverage["coverage_available"]["python"] is False
    assert coverage["coverage_available"]["native"] is False
    assert coverage["main_coverage_label"] == "API execution coverage %"
    assert coverage["line_coverage_percent"] is None
    assert coverage["apis_with_low_coverage"] is None
    assert coverage["per_api"][0]["line_coverage_percent"] is None
    assert coverage["covered_api_count"] == 1
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["apis_with_low_coverage"] is None
    assert "API execution coverage over the accepted API subset" in (results_dir / "coverage_report.md").read_text(encoding="utf-8")
    assert bugs["schema_version"] == "2.0"
    assert bugs["bugs"] == []
    assert bugs["candidate_bugs"] == []
    assert (results_dir / "bug_report.csv").exists()
    assert list((results_dir / "execution_results").glob("*.results.json"))
    assert not list(results_dir.rglob("*.repaired.json"))


def test_stage4_worker_timeout_is_recorded(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(init_dir / "tests.fixtures.fake_stage4_api.sleep_api.init.json", "tests.fixtures.fake_stage4_api.sleep_api", "tests.fixtures.fake_stage4_api.sleep_api")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--worker-timeout-sec",
            "1",
        ],
        check=True,
        env={**os.environ, "DEEPFUZZ_REPORT_BASE_FAILURES_AS_BUGS": "1"},
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    result_payload = json.loads(next((results_dir / "execution_results").glob("*.results.json")).read_text(encoding="utf-8"))
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))

    assert summary["base_fail"] == 1
    assert result_payload["worker_timeout"] is True
    assert bugs["summary"]["total_candidates"] == 1
    assert bugs["bugs"][0]["category"] == "timeout_or_hang"
    assert (results_dir / "bug_report.csv").exists()


def test_stage4_expected_negative_rejection_is_not_bug_or_failure(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(init_dir / "tests.fixtures.fake_stage4_api.int_only.init.json", "tests.fixtures.fake_stage4_api.int_only", "tests.fixtures.fake_stage4_api.int_only")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--mutation-budget",
            "1",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    failures = (results_dir / "failures.csv").read_text(encoding="utf-8")
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))

    assert summary["expected_negative_rejection"] == 1
    assert "type_widen" not in failures
    assert bugs["summary"]["total_candidates"] == 0


def test_stage4_isolated_valid_mutation_abort_becomes_candidate_bug(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(
        init_dir / "tests.fixtures.fake_stage4_api.abort_on_two.init.json",
        "tests.fixtures.fake_stage4_api.abort_on_two",
        "tests.fixtures.fake_stage4_api.abort_on_two",
    )

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--mutation-budget",
            "1",
            "--worker-timeout-sec",
            "20",
            "--case-timeout-sec",
            "5",
        ],
        check=True,
    )

    results = json.loads(next((results_dir / "execution_results").glob("*.results.json")).read_text(encoding="utf-8"))
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))

    assert results["worker_exit_code"] == 0
    assert any(m.get("classification") == "valid_mutation_failure" for m in results["mutations"])
    assert bugs["summary"]["total_candidates"] == 1
    assert bugs["bugs"][0]["category"] == "crash_or_abort"


def test_stage4_bug_report_excludes_final_unresolved_repairs_from_candidates(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    unresolved = tmp_path / "unresolved.csv"
    with unresolved.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["library", "api_full_name", "stage", "error_type", "message", "payload_ref"])
        writer.writeheader()
        writer.writerow({
            "library": "fake",
            "api_full_name": "fake.api",
            "stage": "stage3_init_generation",
            "error_type": "seed_generation_error",
            "message": "insufficient grounded evidence",
            "payload_ref": "fake.init.json",
        })

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--unresolved-failures-csv",
            str(unresolved),
        ],
        check=True,
    )

    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))
    assert bugs["summary"]["unresolved"] == 0
    assert bugs["summary"]["excluded_pipeline_issues"] == 1
    assert bugs["bugs"] == []
    assert bugs["candidate_bugs"] == []
    assert bugs["excluded_pipeline_issues"]["issues"][0]["category"] == "unresolved_pipeline_failure"


def test_stage4_worker_timeout_after_success_is_not_library_bug(tmp_path):
    result_path = tmp_path / "api.results.json"
    init_path = tmp_path / "api.init.json"
    init_path.write_text("{}", encoding="utf-8")
    result_path.write_text(json.dumps({
        "api_full_name": "fake.api",
        "init_json_path": str(init_path),
        "seed": 1,
        "worker_exit_code": -9,
        "worker_timeout": True,
        "base_kwargs_materialized": True,
        "base_execution": {
            "success": True,
            "invoked_api": True,
            "result_summary": {"python_type": "int", "value": 1},
        },
        "mutations": [],
    }), encoding="utf-8")

    report = build_bug_report(
        str(tmp_path),
        [{
            "api_full_name": "fake.api",
            "backend": "fake",
            "library_version": "1.0",
            "init_json_path": str(init_path),
            "results_json_path": str(result_path),
            "worker_exit_code": "-9",
            "worker_timeout": "True",
            "base_success": "True",
        }],
        mutation_budget=1,
    )

    assert report["summary"]["total_candidates"] == 0
    assert report["candidate_bugs"] == []


def test_device_oracle_mismatch_becomes_candidate_bug(tmp_path):
    result_path = tmp_path / "api.results.json"
    init_path = tmp_path / "api.init.json"
    init_path.write_text("{}", encoding="utf-8")
    result_path.write_text(json.dumps({
        "api_full_name": "fake.api",
        "init_json_path": str(init_path),
        "seed": 1,
        "worker_exit_code": 0,
        "worker_timeout": False,
        "base_kwargs_materialized": True,
        "base_execution": {
            "success": True,
            "invoked_api": True,
            "error_type": "DifferentialMismatch",
            "error": "signed zero bits differ",
            "differential_mismatch": True,
            "device_oracle": {
                "enabled": True,
                "available": True,
                "mismatch": True,
                "devices": ["cpu", "cuda:0"],
                "reason": "signed zero bits differ",
            },
            "result_summary": {"python_type": "Tensor", "shape": [1], "dtype": "float32"},
        },
        "mutations": [],
    }), encoding="utf-8")

    report = build_bug_report(
        str(tmp_path),
        [{
            "api_full_name": "fake.api",
            "backend": "fake",
            "library_version": "1.0",
            "init_json_path": str(init_path),
            "results_json_path": str(result_path),
            "worker_exit_code": "0",
            "worker_timeout": "False",
            "base_success": "True",
        }],
        mutation_budget=1,
    )

    assert report["summary"]["total_candidates"] == 1
    assert report["candidate_bugs"][0]["category"] == "differential_mismatch"
    assert report["candidate_bugs"][0]["device_oracle"]["reason"] == "signed zero bits differ"


def test_large_magnitude_edge_mutation_preserves_variation():
    np = pytest.importorskip("numpy")
    arr = np.zeros(4, dtype=np.float16)

    mutated = mutate_value(arr, "value_large_magnitude", random.Random(0))

    assert mutated.dtype == arr.dtype
    assert np.any(mutated > 0)
    assert np.any(mutated < 0)


def test_integer_edge_oracle_rules_do_not_add_float_specials():
    meta = {
        "chosen_type": "int",
        "base_seed_spec": {"kind": "literal", "value": 1},
        "valid_mutation_rules": ["scalar_delta"],
    }

    rules = choose_rules(meta, 4, random.Random(0), "valid", edge_oracle_mutations=True)

    assert "value_large_magnitude" in rules
    assert "value_nan" not in rules
    assert "value_posinf" not in rules


def test_valid_axis_mutations_are_repaired_before_execution(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_payload(
        init_dir / "tests.fixtures.fake_stage4_api.axis_api.init.json",
        "tests.fixtures.fake_stage4_api.axis_api",
        "tests.fixtures.fake_stage4_api.axis_api",
        {
            "x": {
                "name": "x",
                "spec": {"type": "tensor", "flag": "Required", "description": "array"},
                "chosen_type": "tensor",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "tensor", "shape": [2], "dtype": "float32", "fill": 1, "backend": "python"},
                "mutation_rules": ["noop_clone"],
                "valid_mutation_rules": ["noop_clone"],
                "negative_mutation_rules": [],
                "unresolved_reason": "",
            },
            "axes": {
                "name": "axes",
                "spec": {"type": "tuple", "flag": "Required", "description": "unique axes"},
                "chosen_type": "tuple",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "literal", "value": [0], "as_tuple": True},
                "mutation_rules": ["structure_grow", "element_delta"],
                "valid_mutation_rules": ["structure_grow", "element_delta"],
                "negative_mutation_rules": [],
                "unresolved_reason": "",
            },
        },
    )

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "api_only",
            "--mutation-budget",
            "2",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    bundle = json.loads(next((results_dir / "execution_results").glob("*.results.json")).read_text(encoding="utf-8"))
    axis_mutations = [m for m in bundle["mutations"] if m.get("param") == "axes" and m.get("mutation_intent") == "valid"]
    assert axis_mutations
    assert all(m["success"] for m in axis_mutations)
    assert all(m.get("classification") == "valid_mutation_success" for m in axis_mutations)
    assert json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))["invalid_valid_mutation"] == 0


def test_valid_index_and_uint8_mutations_preserve_constraints(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_payload(
        init_dir / "tests.fixtures.fake_stage4_api.index_api.init.json",
        "tests.fixtures.fake_stage4_api.index_api",
        "tests.fixtures.fake_stage4_api.index_api",
        {
            "x": {
                "name": "x",
                "spec": {"type": "tensor", "flag": "Required", "description": "array"},
                "chosen_type": "tensor",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "tensor", "shape": [3], "dtype": "float32", "values": [1, 2, 3], "backend": "python"},
                "mutation_rules": ["noop_clone"],
                "valid_mutation_rules": ["noop_clone"],
                "negative_mutation_rules": [],
                "unresolved_reason": "",
            },
            "indices": {
                "name": "indices",
                "spec": {"type": "tensor", "flag": "Required", "description": "indices"},
                "chosen_type": "tensor",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "tensor", "shape": [1], "dtype": "int64", "fill": 0, "backend": "python"},
                "mutation_rules": ["value_noise", "value_division"],
                "valid_mutation_rules": ["value_noise", "value_division"],
                "negative_mutation_rules": [],
                "unresolved_reason": "",
            },
        },
    )
    _write_payload(
        init_dir / "tests.fixtures.fake_stage4_api.uint8_api.init.json",
        "tests.fixtures.fake_stage4_api.uint8_api",
        "tests.fixtures.fake_stage4_api.uint8_api",
        {
            "a": {
                "name": "a",
                "spec": {"type": "tensor", "flag": "Required", "description": "unsigned byte data"},
                "chosen_type": "tensor",
                "include_in_base_call": True,
                "base_seed_spec": {"kind": "tensor", "shape": [2], "dtype": "uint8", "fill": 2, "backend": "python"},
                "mutation_rules": ["value_noise", "value_division"],
                "valid_mutation_rules": ["value_noise", "value_division"],
                "negative_mutation_rules": [],
                "unresolved_reason": "",
            }
        },
    )

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "api_only",
            "--mutation-budget",
            "2",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))
    assert summary["invalid_valid_mutation"] == 0
    assert bugs["summary"]["total_candidates"] == 0
    for path in (results_dir / "execution_results").glob("*.results.json"):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert all(m["success"] for m in bundle["mutations"] if m.get("mutation_intent") == "valid")


def test_unique_valid_programs_aggregates_from_testcase_hashes(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(init_dir / "tests.fixtures.fake_stage4_api.add_one.init.json", "tests.fixtures.fake_stage4_api.add_one", "tests.fixtures.fake_stage4_api.add_one")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "api_only",
            "--mutation-budget",
            "1",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((results_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert summary["total_valid_programs"] > 0
    assert summary["unique_valid_programs"] > 0
    assert coverage["total_valid_programs"] == summary["total_valid_programs"]
    assert coverage["unique_valid_programs"] == summary["unique_valid_programs"]


def test_python_coverage_combines_isolated_worker_data(tmp_path):
    pytest.importorskip("coverage")
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_init(init_dir / "tests.fixtures.fake_stage4_api.add_one.init.json", "tests.fixtures.fake_stage4_api.add_one", "tests.fixtures.fake_stage4_api.add_one")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "python",
            "--enable-python-coverage",
            "--python-cov-source",
            "tests.fixtures",
            "--mutation-budget",
            "0",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    coverage = json.loads((results_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert coverage["coverage_method"] == "python_coverage"
    assert coverage["coverage_available"]["python"] is True
    assert coverage["python_line_coverage_percent"] is not None
    assert coverage["native_line_coverage_percent"] is None


def test_final_report_helpers_reconcile_bug_and_pipeline_classifications():
    coverage = {
        "coverage_method": "api_coverage_only",
        "api_coverage_percent": 100.0,
        "line_coverage_percent": None,
        "coverage_available": {"api_execution": True, "python": False, "native": False},
    }
    bugs = {
        "candidate_bugs": [],
        "excluded_pipeline_issues": {
            "issues": [
                {
                    "api": "pkg.bad_seed",
                    "stage": "stage3_init_generation",
                    "category": "unresolved_pipeline_failure",
                    "stdout_stderr_excerpt": "seed failed",
                }
            ]
        },
    }
    failures = [
        {
            "api_full_name": "pkg.invalid",
            "stage": "mutation_generator",
            "error_type": "invalid_valid_mutation",
            "error": "axis out of range",
        }
    ]

    pipeline_errors = _stage4_pipeline_errors(failures, coverage, bugs)
    classifications = {row["classification"] for row in pipeline_errors}

    assert "invalid_generated_input" in classifications
    assert "coverage_unavailable" in classifications
    assert "pipeline_error" in classifications
    assert _coverage_payload(coverage)["api_execution_coverage_percent"] == 100.0


def test_target_version_metadata_uses_importable_library_version(tmp_path, monkeypatch):
    (tmp_path / "versionedlib.py").write_text("__version__ = '1.2.3'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    rows = [{"backend": "versionedlib", "library_version": "", "api_full_name": "versionedlib.fn"}]

    assert target_version_from_rows(rows) == "1.2.3"
    meta = collect_environment_metadata("versionedlib", "", seed=7, mutation_budget=3, run_id="r1", coverage_method="api_coverage_only")
    assert meta["target_version"] == "1.2.3"
    assert meta["seed"] == 7
    assert meta["mutation_budget"] == 3


def test_jax_padding_enum_invalid_mutation_is_expected_negative(tmp_path):
    init_dir = tmp_path / "init"
    results_dir = tmp_path / "results"
    init_dir.mkdir()
    _write_padding_init(init_dir / "jax.lax.padtype_to_pads.init.json")

    subprocess.run(
        [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            str(init_dir),
            "--results-dir",
            str(results_dir),
            "--coverage-scope",
            "none",
            "--mutation-budget",
            "2",
            "--worker-timeout-sec",
            "20",
        ],
        check=True,
    )

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    bundle = json.loads(next((results_dir / "execution_results").glob("*.results.json")).read_text(encoding="utf-8"))
    bugs = json.loads((results_dir / "bug_report.json").read_text(encoding="utf-8"))

    assert any(m["rule"] == "enum_switch" and m["success"] for m in bundle["mutations"])
    assert any(m["rule"] == "enum_invalid" and m["classification"] == "expected_negative_rejection" for m in bundle["mutations"])
    assert summary["expected_negative_rejection"] == 1
    assert bugs["summary"]["total_candidates"] == 0


def test_jax_padtype_valid_mut_false_positive_reclassified(tmp_path):
    init_path = tmp_path / "jax.lax.padtype_to_pads.init.json"
    results_path = tmp_path / "jax.lax.padtype_to_pads.results.json"
    _write_padding_init(init_path)
    results_path.write_text(json.dumps({
        "api_full_name": "jax.lax.padtype_to_pads",
        "init_json_path": str(init_path),
        "worker_exit_code": 0,
        "worker_timeout": False,
        "base_execution": {"success": True},
        "mutations": [
            {
                "success": False,
                "error_type": "ValueError",
                "error": "Unknown padding type: VALID_mut",
                "param": "padding",
                "rule": "string_replace",
                "mutation_intent": "valid",
                "expected_failure": False,
            }
        ],
    }), encoding="utf-8")
    bug_report = {
        "schema_version": "2.0",
        "target_library": "jax",
        "target_version": "test",
        "bugs": [
            {
                "bug_id": "bug-old",
                "api": "jax.lax.padtype_to_pads",
                "stage": "mutation",
                "category": "unexpected_exception_on_valid_input",
                "status": "candidate",
                "testcase": str(results_path),
                "signature": "old",
            }
        ],
        "summary": {},
    }

    updated = reclassify_bug_report(bug_report)

    assert updated["bugs"][0]["status"] == "false_positive"
    assert updated["bugs"][0]["category"] == "invalid_generated_mutation"
    assert updated["summary"]["total_candidates"] == 0
