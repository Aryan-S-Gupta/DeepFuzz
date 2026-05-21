#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.model_config import check_model_backend, load_model_config
from common.pipeline_contract import error_record
from common.result_io import (
    atomic_write_csv,
    atomic_write_json,
    discover_libraries,
    read_csv_dicts,
    read_json,
    write_api_list,
)

STAGE_ORDER = {
    "stage1_info2json": 0,
    "stage2_json_validation": 1,
    "stage3_init_generation": 2,
    "stage4_coverage": 3,
    "stage4_runtime": 3,
}

NORMAL_VALIDATION_ERROR_TYPES = {
    "TypeError",
    "ValueError",
    "InvalidArgumentError",
    "NotFoundError",
    "UnimplementedError",
    "OpError",
    "IndexError",
    "KeyError",
}

ERROR_FIELDS = [
    "library",
    "api_full_name",
    "stage",
    "error_type",
    "message",
    "signature",
    "payload_ref",
    "source_file",
    "oracle_decision",
]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _api_from_row(row: Dict[str, Any]) -> str:
    return _text(row.get("api_full_name") or row.get("api") or row.get("name")).strip()


def normalize_error(row: Dict[str, Any], stage: str, source_file: Path, lib: str) -> Dict[str, str]:
    api = _api_from_row(row)
    message = _text(
        row.get("message")
        or row.get("error")
        or row.get("reason")
        or row.get("errors")
        or row.get("stage3_preview_reasons")
        or row.get("category")
    ).strip()
    error_type = _text(
        row.get("error_type")
        or row.get("status")
        or row.get("classification")
        or row.get("stage3_status")
        or row.get("reason")
        or "Error"
    ).strip()
    payload_ref = _text(
        row.get("payload_ref")
        or row.get("json_path")
        or row.get("init_json_path")
        or row.get("results_json_path")
        or row.get("testcase")
    ).strip()
    rec = error_record(api=api, stage=stage, error_type=error_type, message=message, payload_ref=payload_ref)
    return {
        "library": lib,
        "api_full_name": rec["api"],
        "stage": stage,
        "error_type": rec["error_type"],
        "message": rec["message"],
        "signature": rec["signature"],
        "payload_ref": rec["payload_ref"],
        "source_file": str(source_file),
        "oracle_decision": _text(row.get("oracle_decision")),
    }


def is_expected_negative_stage4(row: Dict[str, Any], stage: str) -> bool:
    if stage not in {"stage4_coverage", "stage4_runtime"}:
        return False
    intent = _text(row.get("mutation_intent")).strip().lower()
    error_type = _text(row.get("error_type")).strip()
    classification = _text(row.get("classification")).strip()
    if classification in {"expected_negative_rejection", "negative_accepted_not_bug", "negative_mutation_accepted"}:
        return True
    if error_type in {"expected_negative_rejection", "negative_accepted_not_bug", "negative_mutation_accepted"}:
        return True
    return intent == "negative" and error_type in NORMAL_VALIDATION_ERROR_TYPES


def _csv_sources_for_library(lib: str, stage4_results_dir: str = "") -> List[Tuple[Path, str]]:
    sources: List[Tuple[Path, str]] = [
        (ROOT / "info2json" / "results" / lib / "failures.csv", "stage1_info2json"),
        (ROOT / "json_validator" / "results" / lib / "errors.csv", "stage2_json_validation"),
        (ROOT / "json2init" / "results" / lib / "errors.csv", "stage3_init_generation"),
    ]
    stage4_dirs: List[Path] = []
    if stage4_results_dir:
        stage4_dirs.append(Path(stage4_results_dir))
    stage4_dirs.append(ROOT / "stage4" / "results" / f"{lib}-coverage")
    for directory in stage4_dirs:
        sources.append((directory / "failure.csv", "stage4_runtime"))
        sources.append((directory / "failures.csv", "stage4_runtime"))
        sources.append((directory / "low_coverage.csv", "stage4_coverage"))
        sources.append((directory / "bug_report.csv", "stage4_runtime"))
    return sources


def _coverage_report_errors(lib: str, stage4_results_dir: str = "") -> List[Dict[str, str]]:
    directories = [Path(stage4_results_dir)] if stage4_results_dir else []
    directories.append(ROOT / "stage4" / "results" / f"{lib}-coverage")
    out: List[Dict[str, str]] = []
    for directory in directories:
        report_path = directory / "coverage_report.json"
        report = read_json(report_path, default={})
        if not isinstance(report, dict):
            continue
        threshold_raw = os.environ.get("DEEPFUZZ_FUNCTION_COVERAGE_THRESHOLD", "").strip()
        threshold: Optional[float] = None
        if threshold_raw:
            try:
                threshold = float(threshold_raw)
            except Exception:
                threshold = None
        for item in report.get("per_api", []) or []:
            if not isinstance(item, dict):
                continue
            api = _text(item.get("api")).strip()
            if not api:
                continue
            if item.get("covered") is not True:
                out.append(normalize_error(
                    {
                        "api_full_name": api,
                        "status": "uncovered_api",
                        "message": f"Stage 4 did not cover API; valid_programs={item.get('valid_programs', 0)} bugs={item.get('bugs', 0)}",
                        "payload_ref": str(report_path),
                    },
                    "stage4_coverage",
                    report_path,
                    lib,
                ))
                continue
            function_percent = item.get("function_line_coverage_percent")
            if threshold is not None and function_percent not in {"", None}:
                try:
                    value = float(function_percent)
                except Exception:
                    continue
                if value < threshold:
                    out.append(normalize_error(
                        {
                            "api_full_name": api,
                            "status": "selected_function_low_coverage",
                            "message": (
                                f"selected_function_line_coverage_percent={value:g} below threshold {threshold:g}; "
                                f"missing_lines={item.get('function_missing_lines', [])}"
                            ),
                            "payload_ref": str(report_path),
                        },
                        "stage4_coverage",
                        report_path,
                        lib,
                    ))
    return out


def _low_coverage_rows_applicable(path: Path) -> bool:
    report = read_json(path.parent / "coverage_report.json", default={})
    if not isinstance(report, dict):
        return False
    if report.get("low_coverage_applicability") == "not_applicable_source_line_coverage_unavailable":
        return False
    available = report.get("coverage_available", {})
    if isinstance(available, dict) and not (available.get("python") or available.get("native")):
        return False
    return True


def collect_errors(lib: str, stage4_results_dir: str = "") -> List[Dict[str, str]]:
    dedup: Dict[str, Dict[str, str]] = {}
    for path, stage in _csv_sources_for_library(lib, stage4_results_dir):
        if path.name == "low_coverage.csv" and not _low_coverage_rows_applicable(path):
            continue
        for row in read_csv_dicts(path):
            if path.name == "bug_report.csv":
                status = _text(row.get("status")).strip()
                category = _text(row.get("category")).strip()
                if status == "false_positive" or category in {"invalid_generated_mutation", "expected_exception_on_invalid_input"}:
                    continue
            if is_expected_negative_stage4(row, stage):
                continue
            rec = normalize_error(row, stage, path, lib)
            if not rec["api_full_name"]:
                continue
            key = f"{rec['library']}|{rec['api_full_name']}|{rec['stage']}|{rec['signature']}"
            dedup.setdefault(key, rec)
    for rec in _coverage_report_errors(lib, stage4_results_dir):
        key = f"{rec['library']}|{rec['api_full_name']}|{rec['stage']}|{rec['signature']}"
        dedup.setdefault(key, rec)
    return sorted(dedup.values(), key=lambda r: (r["library"], r["api_full_name"], STAGE_ORDER.get(r["stage"], 99), r["signature"]))


def root_cause_by_api(errors: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in errors:
        api = row.get("api_full_name") or row.get("api") or ""
        if api:
            grouped.setdefault(api, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for api, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: (STAGE_ORDER.get(r.get("stage", ""), 99), r.get("signature", "")))
        out[api] = {
            "root_stage": ordered[0].get("stage", ""),
            "error_count": len(rows),
            "stages": sorted({r.get("stage", "") for r in rows}, key=lambda s: STAGE_ORDER.get(s, 99)),
            "first_error": ordered[0],
        }
    return out


def _run(cmd: Sequence[str]) -> None:
    print("[repair] " + " ".join(subprocess.list2cmdline([x]) if " " in x else x for x in cmd), flush=True)
    subprocess.run(list(cmd), cwd=str(ROOT), check=True)


def _write_errors_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    atomic_write_csv(path, rows, ERROR_FIELDS)


def _selected_apis(errors: Sequence[Dict[str, str]]) -> List[str]:
    return sorted({row["api_full_name"] for row in errors if row.get("api_full_name")})


def _load_api_allowlist(path: str) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def _stage4_dir(lib: str, override: str = "") -> str:
    return override or os.environ.get("STAGE4_RESULTS_DIR", f"stage4/results/{lib}-coverage")


def _write_repair_snapshot(
    repair_dir: Path,
    lib: str,
    errors: Sequence[Dict[str, str]],
    cfg_summary: Dict[str, Any],
    phase: str,
) -> Path:
    repair_dir.mkdir(parents=True, exist_ok=True)
    context_csv = repair_dir / f"{phase}_errors.csv"
    _write_errors_csv(context_csv, errors)
    roots = root_cause_by_api(errors)
    api_list = repair_dir / f"{phase}_api_list.txt"
    apis = write_api_list(api_list, _selected_apis(errors))
    payload = {
        "schema_version": "3.0",
        "library": lib,
        "phase": phase,
        "total_error_records": len(errors),
        "unique_failed_apis": len(apis),
        "root_cause_by_api": roots,
        "by_root_stage": {
            stage: sum(1 for item in roots.values() if item.get("root_stage") == stage)
            for stage in sorted(STAGE_ORDER, key=lambda s: STAGE_ORDER[s])
        },
        "errors_csv": str(context_csv),
        "api_list": str(api_list),
        "model_config": cfg_summary,
    }
    atomic_write_json(repair_dir / f"{phase}_repair_summary.json", payload)
    if phase == "after":
        atomic_write_json(repair_dir / "repair_summary.json", payload)
    return context_csv


def build_oracle_prompt(
    api: str,
    lib: str,
    errors: Sequence[Dict[str, str]],
    spec_path: Path,
    init_path: Path,
    coverage_path: Path,
) -> str:
    spec = read_json(spec_path, default={})
    init = read_json(init_path, default={})
    coverage = read_json(coverage_path, default={})
    relevant = [row for row in errors if row.get("api_full_name") == api]
    snippets = _repair_source_snippets(api)
    return (
        "You are the DeepFuzz repair oracle. Classify the likely failing stage and propose a grounded repair only if the evidence is sufficient.\n"
        "Allowed stages: Stage 1 spec extraction, Stage 2 schema validation, Stage 3 init/seed generation, Stage 4 fuzzing/mutation/coverage, true library/runtime bug.\n"
        "Use only the API name, generated spec JSON, init JSON, validator errors, runtime/fuzzing errors, coverage report, and repository source snippets already provided by the caller.\n"
        "Do not invent API semantics. If evidence is insufficient, return no_patch with a concise reason.\n"
        "Return JSON with keys: decision, likely_stage, grounded_reason, patch_kind, patch.\n\n"
        f"library={lib}\napi={api}\nerrors={json.dumps(relevant, ensure_ascii=False, indent=2)}\n"
        f"spec_json={json.dumps(spec, ensure_ascii=False, indent=2)[:12000]}\n"
        f"init_json={json.dumps(init, ensure_ascii=False, indent=2)[:12000]}\n"
        f"coverage_report={json.dumps(coverage, ensure_ascii=False, indent=2)[:12000]}\n"
        f"source_snippets={json.dumps(snippets, ensure_ascii=False, indent=2)[:12000]}\n"
    )


def _source_snippets_for_file(path: Path, anchors: Sequence[str], window: int = 2400) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks: List[str] = []
    for anchor in anchors:
        idx = text.find(anchor)
        if idx == -1:
            continue
        chunks.append(text[idx: idx + window])
    return "\n\n".join(chunks)


def _repair_source_snippets(api: str) -> Dict[str, str]:
    anchors = {
        "info2json/info2json.py": ["def normalize_schema", "def validate_spec"],
        "json_validator/json_validator.py": ["def validate_normalized_spec", "REPAIR_PROMPT"],
        "json2init/deepfuzz_common.py": ["def build_base_seed_spec", "def build_runtime_object_from_spec", "def classify_smoke_error"],
        "stage4/stage4_coverage_runner.py": ["def build_bug_report", "def build_coverage_report"],
    }
    if api.startswith("tensorflow."):
        anchors["json2init/adapters/tensorflow_adapter.py"] = ["class TensorFlowAdapter"]
    elif api.startswith("torch."):
        anchors["json2init/adapters/torch_adapter.py"] = ["class TorchAdapter"]
    return {
        rel: snippet
        for rel, pats in anchors.items()
        if (snippet := _source_snippets_for_file(ROOT / rel, pats))
    }


def _record_oracle_prompts(lib: str, repair_dir: Path, errors: Sequence[Dict[str, str]], stage4_results_dir: str = "") -> None:
    coverage_path = ROOT / _stage4_dir(lib, stage4_results_dir) / "coverage_report.json"
    for api in _selected_apis(errors):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in api)
        prompt = build_oracle_prompt(
            api=api,
            lib=lib,
            errors=errors,
            spec_path=ROOT / "info2json" / "results" / lib / f"{api}.json",
            init_path=ROOT / "json2init" / "results" / lib / f"{api}.init.json",
            coverage_path=coverage_path,
        )
        (repair_dir / "oracle_prompts").mkdir(parents=True, exist_ok=True)
        (repair_dir / "oracle_prompts" / f"{safe}.prompt.txt").write_text(prompt, encoding="utf-8")


def _append_unresolved_to_bug_report(lib: str, stage4_dir: str, unresolved_csv: Path) -> None:
    try:
        from stage4.stage4_coverage_runner import load_unresolved_failure_bugs, write_bug_report_csv
    except Exception as exc:
        print(f"[repair] warning: could not update bug report with unresolved entries: {exc}", flush=True)
        return
    stage4_path = ROOT / stage4_dir
    bug_path = stage4_path / "bug_report.json"
    bug_report = read_json(bug_path, default={})
    if not isinstance(bug_report, dict):
        bug_report = {"schema_version": "2.0", "target_library": lib, "target_version": "", "bugs": [], "summary": {}}
    target_library = _text(bug_report.get("target_library") or lib)
    target_version = _text(bug_report.get("target_version"))
    additions = load_unresolved_failure_bugs(str(unresolved_csv), target_library, target_version)
    excluded = bug_report.get("excluded_pipeline_issues")
    if not isinstance(excluded, dict):
        excluded = {
            "count": 0,
            "summary": "Pipeline/config/coverage/seed failures are excluded from candidate library bugs.",
            "issues": [],
        }
    issues = excluded.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    by_sig = {str(row.get("signature", "")): row for row in issues if isinstance(row, dict)}
    for issue in additions:
        sig = str(issue.get("signature", ""))
        if sig not in by_sig:
            issues.append(issue)
            by_sig[sig] = issue
    excluded["issues"] = issues[:50]
    excluded["count"] = len(by_sig)
    bug_report["excluded_pipeline_issues"] = excluded
    existing = bug_report.get("bugs", [])
    if not isinstance(existing, list):
        existing = []
    unique = [b for b in existing if isinstance(b, dict) and not b.get("duplicate_of")]
    bug_report["bugs"] = [
        b
        for b in existing
        if isinstance(b, dict) and b.get("category") != "unresolved_pipeline_failure" and b.get("status") != "unresolved"
    ]
    bug_report["candidate_bugs"] = [
        b
        for b in bug_report["bugs"]
        if isinstance(b, dict) and not b.get("duplicate_of") and b.get("status") not in {"unresolved", "false_positive"}
    ]
    summary = bug_report.get("summary") if isinstance(bug_report.get("summary"), dict) else {}
    summary["total_candidates"] = len([b for b in unique if b.get("status") not in {"unresolved", "false_positive"}])
    summary["unresolved"] = 0
    summary["excluded_pipeline_issues"] = excluded["count"]
    summary["false_positives"] = len([b for b in unique if b.get("status") == "false_positive"])
    bug_report["summary"] = summary
    atomic_write_json(bug_path, bug_report)
    write_bug_report_csv(stage4_path / "bug_report.csv", bug_report)


def _reclassify_stage4_bug_report(lib: str, stage4_dir: str) -> Dict[str, Any]:
    try:
        from stage4.stage4_coverage_runner import reclassify_bug_report, write_bug_report_csv
    except Exception as exc:
        print(f"[repair] warning: could not load Stage 4 reclassifier: {exc}", flush=True)
        return {"reclassified": 0, "path": ""}
    stage4_path = ROOT / stage4_dir
    bug_path = stage4_path / "bug_report.json"
    bug_report = read_json(bug_path, default={})
    if not isinstance(bug_report, dict):
        return {"reclassified": 0, "path": str(bug_path)}
    before = sum(1 for b in bug_report.get("bugs", []) if isinstance(b, dict) and b.get("status") == "false_positive")
    updated = reclassify_bug_report(bug_report)
    after = sum(1 for b in updated.get("bugs", []) if isinstance(b, dict) and b.get("status") == "false_positive")
    if after != before:
        atomic_write_json(bug_path, updated)
        write_bug_report_csv(stage4_path / "bug_report.csv", updated)
    return {"reclassified": max(0, after - before), "path": str(bug_path)}


def _run_stage_pipeline(
    lib: str,
    apis: Sequence[str],
    errors_csv: Path,
    api_list_path: Path,
    cfg: Any,
    args: argparse.Namespace,
    force_stage1_apis: Optional[Set[str]] = None,
    stage4_unresolved_csv: str = "",
    mutation_budget: Optional[int] = None,
) -> None:
    if not apis:
        return
    accepted_csv = os.environ.get("ACCEPTED_CSV", f"doc2info/results/{lib}/accepted.csv")
    spec_dir = os.environ.get("SPEC_DIR", f"info2json/results/{lib}")
    json_state_dir = os.environ.get("JSON_STATE_DIR", f"json_validator/results/{lib}")
    init_dir = os.environ.get("INIT_DIR", f"json2init/results/{lib}")
    stage4_dir = _stage4_dir(lib, args.stage4_results_dir)

    if force_stage1_apis:
        gen_list_path = api_list_path.parent / "info2json_api_list.txt"
        write_api_list(gen_list_path, force_stage1_apis)
        _run([
            sys.executable,
            "info2json/info2json.py",
            "--input",
            accepted_csv,
            "--outdir",
            spec_dir,
            "--model",
            cfg.extraction_model,
            "--host",
            cfg.host,
            "--only-api-list",
            str(gen_list_path),
            "--overwrite",
        ])

    _run([
        sys.executable,
        "json_validator/json_validator.py",
        "--spec-dir",
        spec_dir,
        "--api-csv",
        accepted_csv,
        "--state-dir",
        json_state_dir,
        "--primary-repair-model",
        cfg.fast_repair_model,
        "--fallback-repair-model",
        cfg.strong_repair_model,
        "--repair-host",
        cfg.host,
        "--only-apis",
        str(api_list_path),
        "--external-errors-csv",
        str(errors_csv),
    ] + (["--force"] if args.force_validator else []))

    _run([
        sys.executable,
        "json2init/json2init.py",
        "--spec-dir",
        spec_dir,
        "--outdir",
        init_dir,
        "--ok-csv",
        f"{json_state_dir}/ok.csv",
        "--smoke-test",
        "--only-api-list",
        str(api_list_path),
        "--overwrite",
    ])

    if not args.skip_stage4:
        cmd = [
            sys.executable,
            "stage4/stage4_coverage_runner.py",
            "--init-dir",
            init_dir,
            "--results-dir",
            stage4_dir,
            "--ok-csv",
            f"{init_dir}/ok.csv",
            "--coverage-scope",
            os.environ.get("COVERAGE_SCOPE", "both"),
            "--enable-python-coverage",
            "--python-cov-source",
            os.environ.get("PYTHON_COV_SOURCE", lib),
            "--only-api-list",
            str(api_list_path),
            "--mutation-budget",
            str(mutation_budget if mutation_budget is not None else args.mutation_budget),
            "--run-id",
            args.run_id,
            "--merge-unselected",
        ]
        if stage4_unresolved_csv:
            cmd.extend(["--unresolved-failures-csv", stage4_unresolved_csv])
        _run(cmd)


def _repair_one_library(lib: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_model_config()
    run_dir = ROOT / "pipeline_runs" / lib / args.run_id
    repair_dir = run_dir / "repair"
    stage4_reclassification = _reclassify_stage4_bug_report(lib, _stage4_dir(lib, args.stage4_results_dir))
    before_errors = collect_errors(lib, args.stage4_results_dir)
    frozen_apis = _load_api_allowlist(os.environ.get("THESIS_API_LIST", ""))
    if frozen_apis:
        before_errors = [row for row in before_errors if row.get("api_full_name") in frozen_apis]
    before_context = _write_repair_snapshot(repair_dir, lib, before_errors, cfg.to_summary(), "before")
    before_apis = _selected_apis(before_errors)
    api_list_path = repair_dir / "repair_api_list.txt"
    write_api_list(api_list_path, before_apis)

    print(f"[repair] {lib}: collected {len(before_errors)} error records across {len(before_apis)} APIs")
    if args.dry_run or not before_apis:
        return {"library": lib, "before_errors": len(before_errors), "after_errors": len(before_errors), "apis": len(before_apis)}

    roots = root_cause_by_api(before_errors)
    stage1_apis = {api for api, item in roots.items() if item.get("root_stage") == "stage1_info2json"}
    _run_stage_pipeline(
        lib=lib,
        apis=before_apis,
        errors_csv=before_context,
        api_list_path=api_list_path,
        cfg=cfg,
        args=args,
        force_stage1_apis=stage1_apis,
    )
    stage4_reclassification_after_first_run = _reclassify_stage4_bug_report(lib, _stage4_dir(lib, args.stage4_results_dir))

    previous_remaining: Optional[Set[str]] = None
    final_errors = collect_errors(lib, args.stage4_results_dir)
    if frozen_apis:
        final_errors = [row for row in final_errors if row.get("api_full_name") in frozen_apis]
    for iteration in range(1, args.max_repair_iterations + 1):
        remaining = set(_selected_apis(final_errors))
        if not remaining:
            break
        if previous_remaining == remaining:
            print(f"[repair] {lib}: stopping oracle loop; unresolved API set stopped changing", flush=True)
            break
        previous_remaining = set(remaining)

        context_csv = _write_repair_snapshot(repair_dir, lib, final_errors, cfg.to_summary(), f"oracle_round_{iteration}")
        _record_oracle_prompts(lib, repair_dir, final_errors, args.stage4_results_dir)
        ok, message = check_model_backend(cfg.fast_repair_model, cfg.host, backend=cfg.backend)
        if not ok:
            print(f"[repair] {lib}: oracle unavailable: {message}", flush=True)
            for row in final_errors:
                row["oracle_decision"] = "llm_unavailable"
            break

        round_api_list = repair_dir / f"oracle_round_{iteration}_api_list.txt"
        write_api_list(round_api_list, remaining)
        round_mutation_budget = max(args.mutation_budget, args.mutation_budget + iteration * 2)
        _run_stage_pipeline(
            lib=lib,
            apis=sorted(remaining),
            errors_csv=context_csv,
            api_list_path=round_api_list,
            cfg=cfg,
            args=args,
            force_stage1_apis={api for api in remaining if roots.get(api, {}).get("root_stage") == "stage1_info2json"},
            mutation_budget=round_mutation_budget,
        )
        next_errors = collect_errors(lib, args.stage4_results_dir)
        if frozen_apis:
            next_errors = [row for row in next_errors if row.get("api_full_name") in frozen_apis]
        if len(next_errors) >= len(final_errors) and set(_selected_apis(next_errors)) == remaining:
            final_errors = next_errors
            print(f"[repair] {lib}: stopping oracle loop; no measurable improvement", flush=True)
            break
        final_errors = next_errors

    unresolved_csv = repair_dir / "final_unresolved.csv"
    for row in final_errors:
        row.setdefault("oracle_decision", "unresolved_after_bounded_repair")
        if not row.get("oracle_decision"):
            row["oracle_decision"] = "unresolved_after_bounded_repair"
    _write_errors_csv(unresolved_csv, final_errors)
    _write_repair_snapshot(repair_dir, lib, final_errors, cfg.to_summary(), "after")

    if final_errors and not args.skip_stage4:
        _append_unresolved_to_bug_report(lib, _stage4_dir(lib, args.stage4_results_dir), unresolved_csv)
        _reclassify_stage4_bug_report(lib, _stage4_dir(lib, args.stage4_results_dir))

    if not args.skip_report:
        stage4_dir = _stage4_dir(lib, args.stage4_results_dir)
        _run([
            sys.executable,
            "scripts/report.py",
            lib,
            "--run-id",
            args.run_id,
            "--stage4-results-dir",
            stage4_dir,
        ])

    return {
        "library": lib,
        "before_errors": len(before_errors),
        "before_apis": len(before_apis),
        "after_errors": len(final_errors),
        "after_apis": len(_selected_apis(final_errors)),
        "unresolved_csv": str(unresolved_csv),
        "fixed_pipeline_false_positives": int(stage4_reclassification.get("reclassified", 0)) + int(stage4_reclassification_after_first_run.get("reclassified", 0)),
        "unresolved_pipeline_failures": len(final_errors),
        "valid_candidate_bugs": max(0, len(read_csv_dicts(ROOT / _stage4_dir(lib, args.stage4_results_dir) / "bug_report.csv")) - len(final_errors)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Canonical DeepFuzz repair/rerun entry point")
    ap.add_argument("libraries", nargs="*", help="Libraries to repair. Omit to discover all result libraries.")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "latest"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage4-results-dir", default=os.environ.get("STAGE4_RESULTS_DIR", ""))
    ap.add_argument("--skip-stage4", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--force-validator", action="store_true", help="Revalidate APIs already present in Stage 2 ok.csv.")
    ap.add_argument("--max-repair-iterations", type=int, default=int(os.environ.get("DEEPFUZZ_MAX_REPAIR_ITERATIONS", "1")))
    ap.add_argument("--mutation-budget", type=int, default=int(os.environ.get("MUTATION_BUDGET", "8")))
    args = ap.parse_args(argv)

    libraries = args.libraries or discover_libraries(ROOT)
    if libraries == ["all"]:
        libraries = discover_libraries(ROOT)
    if not libraries:
        print("[repair] no result libraries discovered", file=sys.stderr)
        return 1

    summaries = []
    for lib in libraries:
        summaries.append(_repair_one_library(lib, args))
    atomic_write_json(ROOT / "pipeline_runs" / args.run_id / "repair_summary.json", {
        "schema_version": "3.0",
        "libraries": summaries,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
