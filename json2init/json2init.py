
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[0]
for candidate in [Path.cwd(), ROOT, ROOT.parent]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

try:
    from common.result_io import atomic_write_csv
    from json2init.deepfuzz_common import (
        build_runtime_object_from_spec,
        iter_json_specs,
        load_json,
        run_smoke_test,
        write_json,
)
except Exception:
    from common.result_io import atomic_write_csv  # type: ignore
    from deepfuzz_common import (  # type: ignore
        build_runtime_object_from_spec,
        iter_json_specs,
        load_json,
        run_smoke_test,
        write_json,
    )

FIXABLE_SMOKE_CLASSIFICATIONS = {"materialization_error", "spec_or_seed_error", "import_error", "seed_generation_error"}
NON_FIXABLE_SMOKE_CLASSIFICATIONS = {"environment_unsupported", "missing_dependency"}
REPAIRABLE_STAGE3_STATUSES = {"retry", "api_runtime_error"}


def load_ok_api_names(ok_csv: str) -> Set[str]:
    names: Set[str] = set()
    if not ok_csv or not os.path.exists(ok_csv):
        return names
    with open(ok_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            api = str(row.get("api_full_name", "") or "").strip()
            if api:
                names.add(api)
    return names


def load_api_filter(path: str) -> Set[str]:
    names: Set[str] = set()
    if not path or not os.path.exists(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def build_api_full_name(spec: Dict[str, Any], spec_path: str) -> str:
    module_path = str(spec.get("module_path", "") or "").strip()
    api_name = str(spec.get("api_name", "") or "").strip()
    if module_path and api_name:
        return f"{module_path}.{api_name}"
    return Path(spec_path).stem


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    atomic_write_csv(path, rows, fieldnames)


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_rows_by_api(old_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]], selected_apis: Set[str]) -> List[Dict[str, Any]]:
    if not selected_apis:
        return new_rows
    merged: Dict[str, Dict[str, Any]] = {}
    for row in old_rows:
        api = str(row.get("api_full_name", "") or "")
        if api and api not in selected_apis:
            merged[api] = row
    for row in new_rows:
        api = str(row.get("api_full_name", "") or "")
        if api:
            merged[api] = row
    return [merged[k] for k in sorted(merged)]


def classify_stage3_status(spec_ready: bool, smoke_info: Dict[str, Any], smoke_attempted: bool) -> str:
    if not spec_ready:
        return "retry"
    if not smoke_attempted:
        return "spec_ready_no_smoke"
    if smoke_info.get("success"):
        return "ready_for_stage4"
    classification = str(smoke_info.get("classification", "") or "")
    if classification in FIXABLE_SMOKE_CLASSIFICATIONS:
        return "retry"
    if classification == "environment_unsupported":
        return "env_only"
    if classification == "missing_dependency":
        return "missing_dependency"
    if classification == "api_runtime_error":
        return "api_runtime_error"
    return classification or "retry"


def build_issue_reason(runtime_obj: Any, smoke_info: Dict[str, Any], strict_smoke: bool) -> str:
    reasons = list(getattr(runtime_obj, "readiness_reasons", []) or [])
    if reasons:
        return " | ".join(str(x) for x in reasons if str(x))
    classification = str(smoke_info.get("classification", "") or "")
    smoke_error = str(smoke_info.get("error", "") or "")
    if not smoke_info:
        return "not ready"
    if classification in NON_FIXABLE_SMOKE_CLASSIFICATIONS and not strict_smoke:
        return smoke_error or classification or "environment/runtime unavailable"
    return smoke_error or classification or "not ready"


def maybe_attach_smoke_reason(runtime_obj: Any, smoke_info: Dict[str, Any], strict_smoke: bool) -> None:
    if not smoke_info or smoke_info.get("success"):
        return
    classification = str(smoke_info.get("classification", "") or "")
    error_text = str(smoke_info.get("error", "") or classification or "smoke test failed")
    if classification in NON_FIXABLE_SMOKE_CLASSIFICATIONS and not strict_smoke:
        return
    reason = f"smoke test failed: {error_text}"
    reasons = list(getattr(runtime_obj, "readiness_reasons", []) or [])
    if reason not in reasons:
        reasons.append(reason)
    runtime_obj.readiness_reasons = reasons


def _smoke_test_worker(q: Any, api_full_name: str, spec: Dict[str, Any]) -> None:
    try:
        runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
        q.put(run_smoke_test(runtime_obj))
    except BaseException as e:
        q.put({
            "success": False,
            "classification": "api_runtime_error",
            "error": f"smoke worker exception: {type(e).__name__}: {e}",
        })


def safe_run_smoke_test(api_full_name: str, spec: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_smoke_test_worker, args=(q, api_full_name, spec))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.kill()
        p.join()
        return {
            "success": False,
            "classification": "api_runtime_error",
            "error": f"smoke test timed out after {timeout_sec}s",
        }

    if p.exitcode != 0:
        return {
            "success": False,
            "classification": "seed_generation_error",
            "error": f"smoke test subprocess exited abnormally (exitcode={p.exitcode})",
            "exit_code": p.exitcode,
        }

    try:
        out = q.get_nowait()
        return out if isinstance(out, dict) else {
            "success": False,
            "classification": "api_runtime_error",
            "error": "smoke test subprocess returned invalid result",
        }
    except Exception:
        return {
            "success": False,
            "classification": "api_runtime_error",
            "error": "smoke test subprocess returned no result",
        }


def process_specs(
    spec_dir: str,
    outdir: str,
    smoke_test: bool = False,
    overwrite: bool = False,
    reuse_existing: bool = False,
    limit: int = 0,
    ok_csv: Optional[str] = None,
    strict_smoke: bool = False,
    smoke_timeout_sec: int = 30,
    only_api_list: str = "",
) -> None:
    os.makedirs(outdir, exist_ok=True)
    allowed = load_ok_api_names(ok_csv or "") if ok_csv else None
    selected_api_names = load_api_filter(only_api_list)
    previous_stage4_ok = read_csv_rows(os.path.join(outdir, "ok.csv"))
    previous_errors = read_csv_rows(os.path.join(outdir, "errors.csv"))

    stage4_ok_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    summary = {
        "total_seen": 0,
        "total_selected": 0,
        "spec_ready": 0,
        "spec_not_ready": 0,
        "stage3_status_ready_for_stage4": 0,
        "stage3_status_retry": 0,
        "stage3_status_api_runtime_error": 0,
        "stage3_status_env_only": 0,
        "stage3_status_missing_dependency": 0,
        "stage3_status_spec_ready_no_smoke": 0,
        "eligible_for_stage4": 0,
        "not_eligible_for_stage4": 0,
        "repair_candidate": 0,
        "total_valid": 0,
        "total_failed": 0,
        "total_retried": 0,
        "total_repaired": 0,
        "total_unresolved": 0,
        "total_skipped_existing_valid": 0,
        "smoke_success": 0,
        "smoke_materialization_error": 0,
        "smoke_spec_or_seed_error": 0,
        "smoke_import_error": 0,
        "smoke_environment_unsupported": 0,
        "smoke_missing_dependency": 0,
        "smoke_api_runtime_error": 0,
        "smoke_not_attempted": 0,
    }

    status_to_summary_key = {
        "ready_for_stage4": "stage3_status_ready_for_stage4",
        "retry": "stage3_status_retry",
        "api_runtime_error": "stage3_status_api_runtime_error",
        "env_only": "stage3_status_env_only",
        "missing_dependency": "stage3_status_missing_dependency",
        "spec_ready_no_smoke": "stage3_status_spec_ready_no_smoke",
    }

    reached_limit = False
    for spec_path in iter_json_specs(spec_dir):
        summary["total_seen"] += 1
        spec = load_json(spec_path)
        api_full_name = build_api_full_name(spec, spec_path)
        if allowed and api_full_name not in allowed:
            continue
        if selected_api_names and api_full_name not in selected_api_names:
            continue
        if limit and summary["total_selected"] >= limit:
            reached_limit = True
            continue

        summary["total_selected"] += 1
        out_path = os.path.join(outdir, f"{api_full_name}.init.json")
        targeted_overwrite = bool(selected_api_names and api_full_name in selected_api_names)
        if not overwrite and os.path.exists(out_path) and (reuse_existing or not targeted_overwrite):
            try:
                existing = load_json(out_path)
            except Exception:
                existing = {}
            if existing.get("ready_for_stage4"):
                summary["total_skipped_existing_valid"] += 1
                summary["eligible_for_stage4"] += 1
                summary["spec_ready"] += 1
                summary["stage3_status_ready_for_stage4"] += 1
                stage4_ok_rows.append({
                    "api_full_name": api_full_name,
                    "init_json_path": out_path,
                    "runtime_supported_here": str(existing.get("runtime_supported_here", "")),
                    "stage3_status": str(existing.get("stage3_status", "ready_for_stage4")),
                    "smoke_classification": str((existing.get("smoke_test") or {}).get("classification", "")),
                })
                continue
        runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
        smoke_info: Dict[str, Any] = {}
        smoke_attempted = False
        runtime_supported_here: Optional[bool] = None

        if runtime_obj.spec_ready:
            summary["spec_ready"] += 1
        else:
            summary["spec_not_ready"] += 1

        if smoke_test and runtime_obj.spec_ready:
            smoke_attempted = True
            smoke_info = safe_run_smoke_test(api_full_name, spec, timeout_sec=smoke_timeout_sec)
            runtime_obj.smoke_test = smoke_info
            classification = str(smoke_info.get("classification", "") or "")
            if smoke_info.get("success"):
                runtime_supported_here = True
                runtime_obj.ready_for_stage4 = True
                summary["smoke_success"] += 1
            else:
                runtime_supported_here = False
                runtime_obj.ready_for_stage4 = False
                if classification == "materialization_error":
                    summary["smoke_materialization_error"] += 1
                elif classification == "spec_or_seed_error":
                    summary["smoke_spec_or_seed_error"] += 1
                elif classification == "import_error":
                    summary["smoke_import_error"] += 1
                elif classification == "environment_unsupported":
                    summary["smoke_environment_unsupported"] += 1
                elif classification == "missing_dependency":
                    summary["smoke_missing_dependency"] += 1
                else:
                    summary["smoke_api_runtime_error"] += 1
                maybe_attach_smoke_reason(runtime_obj, smoke_info, strict_smoke=strict_smoke)
        else:
            summary["smoke_not_attempted"] += 1
            runtime_obj.ready_for_stage4 = runtime_obj.spec_ready and not smoke_test

        stage3_status = classify_stage3_status(runtime_obj.spec_ready, smoke_info, smoke_attempted)
        summary[status_to_summary_key.get(stage3_status, "stage3_status_retry")] += 1
        if stage3_status in REPAIRABLE_STAGE3_STATUSES:
            summary["repair_candidate"] += 1

        if runtime_obj.ready_for_stage4:
            summary["eligible_for_stage4"] += 1
        else:
            summary["not_eligible_for_stage4"] += 1

        payload = runtime_obj.to_dict()
        payload["runtime_supported_here"] = runtime_supported_here
        payload["source_spec_json_path"] = spec_path
        payload["stage3_status"] = stage3_status
        payload["smoke_attempted"] = smoke_attempted
        payload["repair_candidate"] = stage3_status in REPAIRABLE_STAGE3_STATUSES
        if overwrite or targeted_overwrite or not os.path.exists(out_path):
            write_json(out_path, payload)

        classification = str(smoke_info.get("classification", "") or "")
        reason_text = build_issue_reason(runtime_obj, smoke_info, strict_smoke=strict_smoke)
        materialization_reasons = smoke_info.get("materialization_reasons", []) if isinstance(smoke_info, dict) else []
        if not isinstance(materialization_reasons, list):
            materialization_reasons = []

        if runtime_obj.ready_for_stage4:
            stage4_ok_rows.append({
                "api_full_name": api_full_name,
                "init_json_path": out_path,
                "runtime_supported_here": str(runtime_supported_here),
                "stage3_status": stage3_status,
                "smoke_classification": classification,
            })
        else:
            error_rows.append({
                "api_full_name": api_full_name,
                "stage3_status": stage3_status,
                "smoke_classification": classification,
                "reason": reason_text,
                "init_json_path": out_path,
                "source_spec_json_path": spec_path,
            })

    if selected_api_names:
        stage4_ok_rows = merge_rows_by_api(previous_stage4_ok, stage4_ok_rows, selected_api_names)
        error_rows = merge_rows_by_api(previous_errors, error_rows, selected_api_names)

    write_csv(
        os.path.join(outdir, "ok.csv"),
        stage4_ok_rows,
        ["api_full_name", "init_json_path", "runtime_supported_here", "stage3_status", "smoke_classification"],
    )
    write_csv(
        os.path.join(outdir, "errors.csv"),
        error_rows,
        ["api_full_name", "stage3_status", "smoke_classification", "reason", "init_json_path", "source_spec_json_path"],
    )

    summary["total_valid"] = len(stage4_ok_rows)
    summary["total_failed"] = len(error_rows)
    summary["total_retried"] = len(error_rows)
    summary["total_repaired"] = 0
    summary["total_unresolved"] = len(error_rows)
    print(
        f"[stage3] selected={summary['total_selected']} spec_ready={summary['spec_ready']} "
        f"eligible_for_stage4={summary['eligible_for_stage4']} repair_candidate={summary['repair_candidate']} "
        f"smoke_success={summary['smoke_success']}"
    )
    if reached_limit:
        print(f"[stage3] limit reached: {limit}")
    print(f"[stage3] ok: {os.path.join(outdir, 'ok.csv')}")
    print(f"[stage3] errors: {os.path.join(outdir, 'errors.csv')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: build runtime init objects from validated JSON specs")
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ok-csv", default="", help="Stage 2 ok.csv allowlist")
    ap.add_argument("--smoke-test", action="store_true", help="Run base seed materialization + import + one smoke execution")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true", help="Reuse existing ready *.init.json files where possible")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap after ok/api-list filtering; 0 means no limit")
    ap.add_argument("--smoke-timeout-sec", type=int, default=30)
    ap.add_argument("--only-api-list", default="", help="Optional newline-delimited api_full_name allowlist")
    ap.add_argument(
        "--non-strict-smoke",
        action="store_true",
        help="Do not merge environment-only smoke failures into readiness reasons; still keeps them out of Stage 4 ok.csv.",
    )
    args = ap.parse_args()
    process_specs(
        spec_dir=args.spec_dir,
        outdir=args.outdir,
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
        reuse_existing=args.reuse_existing,
        limit=args.limit,
        ok_csv=args.ok_csv or None,
        strict_smoke=not args.non_strict_smoke,
        smoke_timeout_sec=args.smoke_timeout_sec,
        only_api_list=args.only_api_list,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
