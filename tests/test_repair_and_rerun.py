from __future__ import annotations

import csv
import json

from json_validator.json_validator import process as validate_process
from json_validator.json_validator import validate_normalized_spec
from json2init.json2init import process_specs
from scripts import repair as repair_mod


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validator_preserves_existing_valid_entries(tmp_path):
    spec_dir = tmp_path / "specs"
    state_dir = tmp_path / "state"
    spec_dir.mkdir()
    state_dir.mkdir()
    (spec_dir / "pkg.valid.json").write_text(json.dumps({
        "api_name": "valid",
        "module_path": "pkg",
        "params": {"x": {"type": "int", "flag": "Required", "description": "value", "size": "", "default": "", "constraints": []}},
        "output": {"type": "int", "numbers": "1", "description": "value"},
        "constraints": [],
    }), encoding="utf-8")
    (spec_dir / "pkg.bad.json").write_text(json.dumps({
        "api_name": "bad",
        "module_path": "pkg",
        "params": {},
        "output": {"type": "", "numbers": "", "description": ""},
        "constraints": [],
    }), encoding="utf-8")
    api_csv = tmp_path / "accepted.csv"
    _write_csv(api_csv, ["api_full_name", "api_doc_text"], [
        {"api_full_name": "pkg.valid", "api_doc_text": "Observed signature: (x)\n\nArgs:\n  x (int): value\nReturns:\n  int: value"},
        {"api_full_name": "pkg.bad", "api_doc_text": ""},
    ])
    _write_csv(state_dir / "ok.csv", ["api_full_name", "status", "rounds", "json_path"], [
        {"api_full_name": "pkg.valid", "status": "pass", "rounds": "1", "json_path": "sentinel.json"},
    ])

    validate_process(
        spec_dir=str(spec_dir),
        api_csv=str(api_csv),
        state_dir=str(state_dir),
        primary_repair_model="",
        fallback_repair_model="",
        max_rounds=1,
    )

    ok_rows = list(csv.DictReader((state_dir / "ok.csv").open(encoding="utf-8", newline="")))
    error_rows = list(csv.DictReader((state_dir / "errors.csv").open(encoding="utf-8", newline="")))
    by_api = {row["api_full_name"]: row for row in ok_rows + error_rows}

    assert by_api["pkg.valid"]["json_path"] == "sentinel.json"
    assert by_api["pkg.bad"]["status"] == "fail"
    assert not (state_dir / "validation_report.csv").exists()
    assert not (state_dir / "retry_only.csv").exists()
    assert not (state_dir / "errors.jsonl").exists()


def test_repair_collector_deduplicates_errors_across_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_mod, "ROOT", tmp_path)
    lib = "fake"
    _write_csv(
        tmp_path / "json_validator" / "results" / lib / "errors.csv",
        ["api_full_name", "status", "errors", "json_path"],
        [
            {"api_full_name": "pkg.bad", "status": "retry", "errors": "missing type", "json_path": "pkg.bad.json"},
            {"api_full_name": "pkg.bad", "status": "retry", "errors": "missing type", "json_path": "pkg.bad.json"},
        ],
    )
    _write_csv(
        tmp_path / "json2init" / "results" / lib / "errors.csv",
        ["api_full_name", "stage3_status", "reason", "init_json_path"],
        [{"api_full_name": "pkg.init_bad", "stage3_status": "retry", "reason": "boom", "init_json_path": "pkg.init_bad.init.json"}],
    )

    errors = repair_mod.collect_errors(lib)

    assert [row["api_full_name"] for row in errors] == ["pkg.bad", "pkg.init_bad"]
    assert {row["stage"] for row in errors} == {"stage2_json_validation", "stage3_init_generation"}


def test_validator_classifies_zero_arg_and_prose_removed_cases():
    zero = validate_normalized_spec(
        "pkg.noarg",
        {"api_name": "noarg", "module_path": "pkg", "params": {}, "output": {"type": "int", "numbers": "1", "description": "value"}, "constraints": []},
        "noarg()\nReturns:\n  int: value",
        signature="()",
    )
    prose = validate_normalized_spec(
        "pkg.bad",
        {"api_name": "bad", "module_path": "pkg", "params": {}, "output": {"type": "", "numbers": "", "description": ""}, "constraints": []},
        "Args:\n  Caution: this is not a parameter",
        original_spec={"params": {"Caution": {"type": "if true", "flag": "Required"}}},
    )

    assert zero.status == "zero_arg_valid_api"
    assert prose.status == "all_params_removed_due_to_prose"


def test_repair_collector_ignores_expected_negative_stage4_rejections(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_mod, "ROOT", tmp_path)
    lib = "fake"
    _write_csv(
        tmp_path / "stage4" / "results" / f"{lib}-coverage" / "failures.csv",
        ["api_full_name", "stage", "mutation_intent", "param", "rule", "error_type", "error", "results_json_path"],
        [
            {
                "api_full_name": "pkg.expected",
                "stage": "mutation",
                "mutation_intent": "negative",
                "param": "x",
                "rule": "type_widen",
                "error_type": "InvalidArgumentError",
                "error": "expected invalid dtype",
                "results_json_path": "x.json",
            },
            {
                "api_full_name": "pkg.valid_bad",
                "stage": "mutation_generator",
                "mutation_intent": "valid",
                "param": "x",
                "rule": "scalar_delta",
                "error_type": "invalid_valid_mutation",
                "error": "bad valid mutation",
                "results_json_path": "y.json",
            },
        ],
    )

    errors = repair_mod.collect_errors(lib)

    assert [row["api_full_name"] for row in errors] == ["pkg.valid_bad"]


def test_repair_reclassifies_existing_jax_padding_false_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_mod, "ROOT", tmp_path)
    lib = "jax"
    stage4_dir = tmp_path / "stage4" / "results" / f"{lib}-coverage"
    execution_dir = stage4_dir / "execution_results"
    init_dir = tmp_path / "json2init" / "results" / lib
    execution_dir.mkdir(parents=True)
    init_dir.mkdir(parents=True)
    init_path = init_dir / "jax.lax.padtype_to_pads.init.json"
    results_path = execution_dir / "jax.lax.padtype_to_pads.results.json"
    init_path.write_text(json.dumps({
        "api_full_name": "jax.lax.padtype_to_pads",
        "params": {
            "padding": {
                "spec": {
                    "enum_values": ["VALID", "SAME", "SAME_LOWER"],
                    "case_insensitive": True,
                }
            }
        },
    }), encoding="utf-8")
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
            }
        ],
    }), encoding="utf-8")
    (stage4_dir / "bug_report.json").write_text(json.dumps({
        "schema_version": "2.0",
        "target_library": "jax",
        "target_version": "0.test",
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
        "summary": {"total_candidates": 1},
    }), encoding="utf-8")

    summary = repair_mod._reclassify_stage4_bug_report(lib, f"stage4/results/{lib}-coverage")
    updated = json.loads((stage4_dir / "bug_report.json").read_text(encoding="utf-8"))

    assert summary["reclassified"] == 1
    assert updated["bugs"][0]["status"] == "false_positive"
    assert updated["bugs"][0]["category"] == "invalid_generated_mutation"
    assert updated["summary"]["total_candidates"] == 0
    assert repair_mod.collect_errors(lib) == []


def test_stage3_uses_errors_csv_and_overwrites_selected_init(tmp_path):
    spec_dir = tmp_path / "specs"
    outdir = tmp_path / "init"
    spec_dir.mkdir()
    outdir.mkdir()
    api = "tests.fixtures.fake_stage4_api.add_one"
    (spec_dir / f"{api}.json").write_text(json.dumps({
        "api_name": "add_one",
        "module_path": "tests.fixtures.fake_stage4_api",
        "params": {"x": {"type": "int", "flag": "Required", "description": "integer input", "constraints": []}},
        "output": {"type": "int", "numbers": "1", "description": "result"},
        "constraints": [],
    }), encoding="utf-8")
    _write_csv(tmp_path / "stage2_ok.csv", ["api_full_name"], [{"api_full_name": api}])
    selected = tmp_path / "selected.txt"
    selected.write_text(api + "\n", encoding="utf-8")
    stale = outdir / f"{api}.init.json"
    stale.write_text(json.dumps({"api_full_name": api, "ready_for_stage4": False, "sentinel": "old"}), encoding="utf-8")

    process_specs(
        spec_dir=str(spec_dir),
        outdir=str(outdir),
        ok_csv=str(tmp_path / "stage2_ok.csv"),
        only_api_list=str(selected),
        smoke_test=False,
    )

    rewritten = json.loads(stale.read_text(encoding="utf-8"))
    assert rewritten["ready_for_stage4"] is True
    assert "sentinel" not in rewritten
    assert (outdir / "ok.csv").exists()
    assert (outdir / "errors.csv").exists()
    assert not (outdir / "spec_ok.csv").exists()
    assert not (outdir / "retry_only.csv").exists()
    assert not (outdir / "issues.jsonl").exists()
    assert not (outdir / "errors.jsonl").exists()
