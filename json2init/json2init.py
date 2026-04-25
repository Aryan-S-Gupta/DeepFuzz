
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path(__file__).resolve().parents[0]
for candidate in [Path.cwd(), ROOT, ROOT.parent]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

try:
    from json2init.deepfuzz_common import (
        build_runtime_object_from_spec,
        iter_json_specs,
        load_json,
        run_smoke_test,
        write_json,
    )
except Exception:
    from deepfuzz_common import (  # type: ignore
        build_runtime_object_from_spec,
        iter_json_specs,
        load_json,
        run_smoke_test,
        write_json,
    )

FIXABLE_SMOKE_CLASSIFICATIONS = {"materialization_error", "spec_or_seed_error", "import_error"}
NON_FIXABLE_SMOKE_CLASSIFICATIONS = {"environment_unsupported", "missing_dependency"}
REPAIRABLE_STAGE3_STATUSES = {"retry", "api_runtime_error"}
_ILLEGAL_XLSX_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


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
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _clean_xlsx_value(v: Any) -> Any:
    if isinstance(v, str):
        return _ILLEGAL_XLSX_CHARS_RE.sub("", v)
    return v


def write_xlsx(path: str, rows: List[Dict[str, Any]]) -> None:
    if pd is None:
        return
    ensure_parent(path)
    try:
        df = pd.DataFrame(rows)
        df = df.apply(lambda col: col.map(_clean_xlsx_value))
        df.to_excel(path, index=False)
    except Exception as e:
        print(f"[stage3] skipped xlsx write: {e}", file=sys.stderr)


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
            "classification": "api_runtime_error",
            "error": f"smoke test subprocess exited abnormally (exitcode={p.exitcode})",
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
    ok_csv: Optional[str] = None,
    strict_smoke: bool = False,
    smoke_timeout_sec: int = 30,
    only_api_list: str = "",
) -> None:
    os.makedirs(outdir, exist_ok=True)
    allowed = load_ok_api_names(ok_csv or "") if ok_csv else None
    selected_api_names = load_api_filter(only_api_list)

    stage4_ok_rows: List[Dict[str, Any]] = []
    spec_ok_rows: List[Dict[str, Any]] = []
    retry_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

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

    for spec_path in iter_json_specs(spec_dir):
        summary["total_seen"] += 1
        spec = load_json(spec_path)
        api_full_name = build_api_full_name(spec, spec_path)
        if allowed and api_full_name not in allowed:
            continue
        if selected_api_names and api_full_name not in selected_api_names:
            continue

        summary["total_selected"] += 1
        runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
        smoke_info: Dict[str, Any] = {}
        smoke_attempted = False
        runtime_supported_here: Optional[bool] = None

        if runtime_obj.spec_ready:
            summary["spec_ready"] += 1
            spec_ok_rows.append({
                "api_full_name": api_full_name,
                "source_spec_json_path": spec_path,
            })
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

        out_path = os.path.join(outdir, f"{api_full_name}.init.json")
        payload = runtime_obj.to_dict()
        payload["runtime_supported_here"] = runtime_supported_here
        payload["source_spec_json_path"] = spec_path
        payload["stage3_status"] = stage3_status
        payload["smoke_attempted"] = smoke_attempted
        payload["repair_candidate"] = stage3_status in REPAIRABLE_STAGE3_STATUSES
        if overwrite or not os.path.exists(out_path):
            write_json(out_path, payload)

        classification = str(smoke_info.get("classification", "") or "")
        reason_text = build_issue_reason(runtime_obj, smoke_info, strict_smoke=strict_smoke)
        materialization_reasons = smoke_info.get("materialization_reasons", []) if isinstance(smoke_info, dict) else []
        if not isinstance(materialization_reasons, list):
            materialization_reasons = []

        issue = {
            "api_full_name": api_full_name,
            "stage3_status": stage3_status,
            "repair_candidate": stage3_status in REPAIRABLE_STAGE3_STATUSES,
            "spec_ready": runtime_obj.spec_ready,
            "ready_for_stage4": runtime_obj.ready_for_stage4,
            "runtime_supported_here": runtime_supported_here,
            "reasons": list(runtime_obj.readiness_reasons),
            "reason": reason_text,
            "smoke_test": smoke_info,
            "smoke_attempted": smoke_attempted,
            "smoke_classification": classification,
            "materialization_reasons": materialization_reasons,
            "init_json_path": out_path,
            "source_spec_json_path": spec_path,
        }
        issues.append(issue)

        report_row = {
            "api_full_name": api_full_name,
            "stage3_status": stage3_status,
            "repair_candidate": stage3_status in REPAIRABLE_STAGE3_STATUSES,
            "spec_ready": runtime_obj.spec_ready,
            "ready_for_stage4": runtime_obj.ready_for_stage4,
            "runtime_supported_here": runtime_supported_here,
            "smoke_attempted": smoke_attempted,
            "smoke_success": bool(smoke_info.get("success")) if smoke_info else False,
            "smoke_classification": classification,
            "reason": reason_text,
            "readiness_reasons": " | ".join(str(x) for x in (runtime_obj.readiness_reasons or [])),
            "materialization_reasons": " | ".join(str(x) for x in materialization_reasons),
            "source_spec_json_path": spec_path,
            "init_json_path": out_path,
        }
        report_rows.append(report_row)

        if runtime_obj.ready_for_stage4:
            stage4_ok_rows.append({
                "api_full_name": api_full_name,
                "init_json_path": out_path,
                "runtime_supported_here": str(runtime_supported_here),
                "stage3_status": stage3_status,
                "smoke_classification": classification,
            })
        elif stage3_status in REPAIRABLE_STAGE3_STATUSES:
            retry_rows.append({
                "api_full_name": api_full_name,
                "stage3_status": stage3_status,
                "smoke_classification": classification,
                "reason": reason_text,
                "init_json_path": out_path,
                "source_spec_json_path": spec_path,
            })
            error_rows.append({
                "api_full_name": api_full_name,
                "stage3_status": stage3_status,
                "smoke_classification": classification,
                "reason": reason_text,
                "init_json_path": out_path,
                "source_spec_json_path": spec_path,
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

    write_csv(
        os.path.join(outdir, "ok.csv"),
        stage4_ok_rows,
        ["api_full_name", "init_json_path", "runtime_supported_here", "stage3_status", "smoke_classification"],
    )
    write_csv(
        os.path.join(outdir, "spec_ok.csv"),
        spec_ok_rows,
        ["api_full_name", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "retry_only.csv"),
        retry_rows,
        ["api_full_name", "stage3_status", "smoke_classification", "reason", "init_json_path", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "errors.csv"),
        error_rows,
        ["api_full_name", "stage3_status", "smoke_classification", "reason", "init_json_path", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "validation_report.csv"),
        report_rows,
        [
            "api_full_name",
            "stage3_status",
            "repair_candidate",
            "spec_ready",
            "ready_for_stage4",
            "runtime_supported_here",
            "smoke_attempted",
            "smoke_success",
            "smoke_classification",
            "reason",
            "readiness_reasons",
            "materialization_reasons",
            "source_spec_json_path",
            "init_json_path",
        ],
    )
    write_xlsx(os.path.join(outdir, "validation_report.xlsx"), report_rows)

    with open(os.path.join(outdir, "issues.jsonl"), "w", encoding="utf-8") as f:
        for row in issues:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retry_api_names = sorted({str(r["api_full_name"]) for r in retry_rows})
    with open(os.path.join(outdir, "retry_api_list.txt"), "w", encoding="utf-8") as f:
        for api_name in retry_api_names:
            f.write(api_name + "\n")

    write_json(os.path.join(outdir, "summary.json"), summary)
    print(
        f"[stage3] selected={summary['total_selected']} spec_ready={summary['spec_ready']} "
        f"eligible_for_stage4={summary['eligible_for_stage4']} repair_candidate={summary['repair_candidate']} "
        f"smoke_success={summary['smoke_success']}"
    )
    print(f"[stage3] ok: {os.path.join(outdir, 'ok.csv')}")
    print(f"[stage3] spec_ok: {os.path.join(outdir, 'spec_ok.csv')}")
    print(f"[stage3] retry_only: {os.path.join(outdir, 'retry_only.csv')}")
    print(f"[stage3] errors: {os.path.join(outdir, 'errors.csv')}")
    print(f"[stage3] report: {os.path.join(outdir, 'validation_report.csv')}")
    if pd is not None:
        print(f"[stage3] report_xlsx: {os.path.join(outdir, 'validation_report.xlsx')}")
    print(f"[stage3] summary: {os.path.join(outdir, 'summary.json')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: build runtime init objects from validated JSON specs")
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ok-csv", default="", help="Stage 2 ok.csv allowlist")
    ap.add_argument("--smoke-test", action="store_true", help="Run base seed materialization + import + one smoke execution")
    ap.add_argument("--overwrite", action="store_true")
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
        ok_csv=args.ok_csv or None,
        strict_smoke=not args.non_strict_smoke,
        smoke_timeout_sec=args.smoke_timeout_sec,
        only_api_list=args.only_api_list,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
