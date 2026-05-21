#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import importlib
import json
import os
import platform
import shlex
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
for candidate in [Path.cwd(), ROOT]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

from common.result_io import atomic_write_csv  # noqa: E402
from common.result_io import atomic_write_text  # noqa: E402
from common.api_policy import is_internal_api  # noqa: E402
from json2init.deepfuzz_common import load_json, write_json  # noqa: E402

try:
    from coverage import Coverage  # type: ignore
except Exception:  # pragma: no cover
    Coverage = None  # type: ignore


API_ONLY_SCOPES = {"none", "api_only"}


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in str(name))


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    atomic_write_csv(path, rows, fieldnames)


def read_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    if not os.path.exists(str(path)):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_rows_preserving_unselected(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    selected_apis: Set[str],
    key_fields: Sequence[str] = ("api_full_name",),
) -> List[Dict[str, Any]]:
    if not selected_apis:
        return list(new_rows)
    merged: List[Dict[str, Any]] = []
    for row in old_rows:
        api = str(row.get("api_full_name", "") or "")
        if api and api not in selected_apis:
            merged.append(dict(row))
        elif api == "__CAMPAIGN__":
            continue
    existing_keys = {tuple(str(r.get(k, "")) for k in key_fields) for r in merged}
    for row in new_rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        merged.append(dict(row))
    return sorted(merged, key=lambda r: tuple(str(r.get(k, "")) for k in key_fields))


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def recompute_summary_from_merged_rows(summary: Dict[str, Any], results_rows: Sequence[Dict[str, Any]], failure_rows: Sequence[Dict[str, Any]]) -> None:
    summary["selected_apis"] = len(results_rows)
    summary["executed_apis"] = len(results_rows)
    summary["base_success"] = sum(1 for row in results_rows if _boolish(row.get("base_success")))
    summary["base_fail"] = len(results_rows) - int(summary["base_success"])
    for key in ["mutation_cases", "mutation_success", "mutation_fail", "negative_expected_fail", "expected_negative_rejection", "negative_accepted_not_bug", "invalid_valid_mutation", "device_oracle_mismatch", "valid_programs"]:
        total = 0
        for row in results_rows:
            try:
                total += int(float(str(row.get(key, 0) or 0)))
            except Exception:
                pass
        summary[key] = total
    summary["total_valid_programs"] = int(summary.get("valid_programs", 0) or 0)
    hashes: Set[str] = set()
    for row in results_rows:
        results_path = str(row.get("results_json_path", "") or "")
        if results_path and os.path.exists(results_path):
            try:
                bundle = load_json(results_path)
                hashes.update(str(x) for x in bundle.get("valid_testcase_hashes", []) if str(x))
            except Exception:
                pass
    summary["unique_valid_programs"] = len(hashes) if hashes else int(summary.get("total_valid_programs", 0) or 0)
    summary["apis_with_failures"] = len({str(row.get("api_full_name", "")) for row in failure_rows if str(row.get("api_full_name", ""))})
    summary["apis_with_nan"] = sum(1 for row in results_rows if _boolish(row.get("nan_observed")))


def load_ok_init_paths(ok_csv: str) -> Set[str]:
    paths: Set[str] = set()
    if not ok_csv or not os.path.exists(ok_csv):
        return paths
    with open(ok_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            p = str(row.get("init_json_path", "") or "").strip()
            if p:
                paths.add(os.path.abspath(p))
    return paths


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


def api_name_from_init(path: str) -> str:
    try:
        obj = load_json(path)
        return str(obj.get("api_full_name", "") or Path(path).stem.replace(".init", ""))
    except Exception:
        return Path(path).stem.replace(".init", "")


def iter_init_objects(init_dir: str, ok_csv: str = "", only_api_list: str = "") -> Iterable[str]:
    allow_paths = load_ok_init_paths(ok_csv) if ok_csv else None
    allow_apis = load_api_filter(only_api_list) if only_api_list else None
    env_retry = os.environ.get("STAGE4_RETRY_LIST", "").strip()
    if env_retry and not only_api_list:
        allow_apis = load_api_filter(env_retry)
    for path in sorted(Path(init_dir).glob("*.init.json")):
        full = os.path.abspath(str(path))
        api_full_name = api_name_from_init(str(path))
        if os.environ.get("DEEPFUZZ_INCLUDE_INTERNAL_APIS", "").strip().lower() not in {"1", "true", "yes"}:
            if is_internal_api(api_full_name):
                continue
        if allow_paths is not None and allow_paths and full not in allow_paths:
            continue
        if allow_apis is not None and allow_apis and api_full_name not in allow_apis:
            continue
        yield str(path)


class NativeCoverageSession:
    def __init__(
        self,
        enabled: bool,
        engine: str,
        source_root: str,
        build_dir: str,
        gcovr_executable: str = "gcovr",
        gcov_executable: str = "",
        gcovr_filter: Sequence[str] | None = None,
        gcovr_exclude: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
        emit_html: bool = False,
        gcovr_jobs: int = 1,
    ) -> None:
        self.enabled = bool(enabled and engine == "gcovr")
        self.engine = engine
        self.source_root = os.path.abspath(source_root) if source_root else ""
        self.build_dir = os.path.abspath(build_dir) if build_dir else ""
        self.gcovr_executable = gcovr_executable
        self.gcov_executable = gcov_executable
        self.gcovr_filter = list(gcovr_filter or [])
        self.gcovr_exclude = list(gcovr_exclude or [])
        self.extra_args = list(extra_args or [])
        self.emit_html = bool(emit_html)
        self.gcovr_jobs = max(int(gcovr_jobs or 1), 1)

    def validate_or_raise(self) -> None:
        if not self.enabled:
            return
        if not self.source_root or not os.path.isdir(self.source_root):
            raise RuntimeError(f"native coverage source root does not exist: {self.source_root}")
        if not self.build_dir or not os.path.isdir(self.build_dir):
            raise RuntimeError(f"native coverage build dir does not exist: {self.build_dir}")
        if shutil.which(self.gcovr_executable) is None:
            raise RuntimeError(f"gcovr executable not found: {self.gcovr_executable}")
        if not any(Path(self.build_dir).rglob("*.gcno")):
            raise RuntimeError(
                f"no .gcno files found under {self.build_dir}; build the library with gcc/gcov coverage flags first"
            )

    def reset(self) -> None:
        if not self.enabled:
            return
        for pattern in ("*.gcda", "*.profraw", "*.profdata"):
            for path in Path(self.build_dir).rglob(pattern):
                try:
                    path.unlink()
                except Exception:
                    pass

    def export(self, out_prefix: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        out_json = os.path.abspath(f"{out_prefix}.gcovr_summary.json")
        out_html = os.path.abspath(f"{out_prefix}.gcovr.html")
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        summary: Dict[str, Any] = {
            "enabled": True,
            "engine": "gcovr",
            "summary_json": out_json,
            "html_path": out_html if self.emit_html else "",
        }
        if not any(Path(self.build_dir).rglob("*.gcda")):
            summary["export_error"] = f"no .gcda files found under {self.build_dir} after worker exit"
            return summary
        base_cmd = [
            self.gcovr_executable,
            "-r",
            self.source_root,
            "--object-directory",
            self.build_dir,
            "-j",
            str(self.gcovr_jobs),
            "--json-summary-pretty",
            "--json-summary",
            out_json,
        ]
        if self.gcov_executable:
            base_cmd += ["--gcov-executable", self.gcov_executable]
        for filt in self.gcovr_filter:
            base_cmd += ["--filter", filt]
        for exc in self.gcovr_exclude:
            base_cmd += ["--exclude", exc]
        base_cmd += list(self.extra_args)
        try:
            subprocess.run(base_cmd, cwd=self.build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            summary["export_error"] = exc.stderr or exc.stdout or str(exc)
            return summary
        if self.emit_html:
            html_cmd = [
                self.gcovr_executable,
                "-r",
                self.source_root,
                "--object-directory",
                self.build_dir,
                "-j",
                str(self.gcovr_jobs),
                "--html",
                "--html-details",
                "-o",
                out_html,
            ]
            if self.gcov_executable:
                html_cmd += ["--gcov-executable", self.gcov_executable]
            for filt in self.gcovr_filter:
                html_cmd += ["--filter", filt]
            for exc in self.gcovr_exclude:
                html_cmd += ["--exclude", exc]
            html_cmd += list(self.extra_args)
            try:
                subprocess.run(html_cmd, cwd=self.build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except subprocess.CalledProcessError as exc:
                summary["html_error"] = exc.stderr or exc.stdout or str(exc)
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in [
                "line_percent",
                "line_covered",
                "line_total",
                "branch_percent",
                "branch_covered",
                "branch_total",
                "function_percent",
                "function_covered",
                "function_total",
            ]:
                summary[key] = data.get(key, "")
        except Exception as exc:
            summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        return summary


def numeric_or_none(value: Any) -> Optional[float]:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def evaluate_low_coverage(summary: Dict[str, Any], threshold: float) -> Tuple[bool, str]:
    candidates = [
        ("native_line_percent", summary.get("native_line_percent")),
        ("native_branch_percent", summary.get("native_branch_percent")),
        ("native_function_percent", summary.get("native_function_percent")),
        ("python_statement_percent", summary.get("python_statement_percent")),
        ("python_branch_percent", summary.get("python_branch_percent")),
    ]
    seen_any = False
    for name, raw in candidates:
        value = numeric_or_none(raw)
        if value is None:
            continue
        seen_any = True
        if value < threshold:
            return True, f"{name}={value:.4g} below threshold {threshold:.4g}"
    return (False, "") if seen_any else (False, "")


def compute_bundle_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
    muts = bundle.get("mutations", []) if isinstance(bundle.get("mutations"), list) else []
    invalid_valid_mutations = [
        m
        for m in muts
        if isinstance(m, dict) and not m.get("success") and str(m.get("classification", "")) == "invalid_valid_mutation"
    ]
    unexpected_mut_fail = [
        m
        for m in muts
        if isinstance(m, dict)
        and not m.get("success")
        and not m.get("expected_failure")
        and str(m.get("classification", "")) != "invalid_valid_mutation"
    ]
    valid_mutation_success = [
        m
        for m in muts
        if isinstance(m, dict) and m.get("success") and str(m.get("mutation_intent", "valid")) == "valid"
    ]
    base_valid = bool(base.get("success") and base.get("invoked_api", True))
    valid_hashes: List[str] = []
    if base_valid and base.get("testcase_hash"):
        valid_hashes.append(str(base.get("testcase_hash")))
    for mutation in valid_mutation_success:
        if mutation.get("testcase_hash"):
            valid_hashes.append(str(mutation.get("testcase_hash")))
    if not valid_hashes:
        valid_hashes = [str(x) for x in bundle.get("valid_testcase_hashes", []) if str(x)]
    nan_or_inf = (
        bool(base.get("has_nan"))
        or bool(base.get("has_inf"))
        or any(isinstance(m, dict) and (m.get("has_nan") or m.get("has_inf")) for m in muts)
    )
    device_mismatches = int(bool(base.get("differential_mismatch"))) + sum(
        1 for m in muts if isinstance(m, dict) and bool(m.get("differential_mismatch"))
    )
    valid_programs = (1 if base_valid else 0) + len(valid_mutation_success)
    return {
        "base_success": bool(base.get("success")),
        "mutation_count": len(muts),
        "mutation_success": sum(1 for m in muts if isinstance(m, dict) and m.get("success")),
        "mutation_fail": len(unexpected_mut_fail),
        "expected_negative_rejection": sum(1 for m in muts if isinstance(m, dict) and str(m.get("classification", "")) == "expected_negative_rejection"),
        "negative_accepted_not_bug": sum(1 for m in muts if isinstance(m, dict) and str(m.get("classification", "")) == "negative_accepted_not_bug"),
        "negative_expected_fail": sum(1 for m in muts if isinstance(m, dict) and m.get("expected_failure")),
        "invalid_valid_mutation": len(invalid_valid_mutations),
        "device_oracle_mismatch": device_mismatches,
        "nan_observed": nan_or_inf,
        "valid_programs": valid_programs,
        "valid_testcase_hashes": list(dict.fromkeys(valid_hashes)),
    }


def worker_failure_bundle(init_path: str, seed: int, exit_code: int, timeout: bool, error: str) -> Dict[str, Any]:
    api_full_name = api_name_from_init(init_path)
    try:
        init_obj = load_json(init_path)
    except Exception:
        init_obj = {}
    valid_seed = bool(init_obj.get("ready_for_stage4", False))
    return {
        "api_full_name": api_full_name,
        "import_path": str(init_obj.get("import_path", "") or ""),
        "backend": str(init_obj.get("backend", "") or ""),
        "library_version": "",
        "init_json_path": init_path,
        "seed": seed,
        "base_execution": {
            "success": False,
            "error_type": "WorkerTimeout" if timeout else "WorkerCrash",
            "error": error,
            "result_summary": {},
            "duration_ms": 0.0,
            "has_nan": False,
            "has_inf": False,
            "invoked_api": valid_seed,
        },
        "mutations": [],
        "materialization_errors": [],
        "base_kwargs_materialized": valid_seed,
        "worker_exit_code": exit_code,
        "worker_timeout": timeout,
    }


def launch_worker(
    init_paths: Sequence[str],
    output_json: str,
    mutation_budget: int,
    seed: int,
    timeout_sec: int,
    enable_python_coverage: bool = False,
    python_cov_source: Sequence[str] = (),
    python_cov_omit: Sequence[str] = (),
    python_cov_dir: str = "",
    case_timeout_sec: int = 30,
    enable_device_oracle: bool = False,
    device_oracle_devices: Sequence[str] = (),
    device_oracle_rtol: float = 1e-4,
    device_oracle_atol: float = 1e-5,
    edge_oracle_mutations: bool = False,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "stage4.stage4_worker",
        "--results-json",
        output_json,
        "--mutation-budget",
        str(mutation_budget),
        "--seed",
        str(seed),
        "--case-timeout-sec",
        str(case_timeout_sec),
    ]
    if len(init_paths) == 1:
        cmd += ["--init-path", init_paths[0]]
    else:
        init_list = output_json + ".init_list.txt"
        with open(init_list, "w", encoding="utf-8") as f:
            for path in init_paths:
                f.write(path + "\n")
        cmd += ["--init-list", init_list]
    if enable_python_coverage:
        cmd += [
            "--enable-python-coverage",
            "--python-cov-data-file",
            os.path.join(python_cov_dir, ".coverage"),
            "--python-cov-json",
            os.path.join(python_cov_dir, "coverage.json"),
            "--python-cov-html",
            os.path.join(python_cov_dir, "html"),
        ]
        for src in python_cov_source:
            cmd += ["--python-cov-source", src]
        for omit in python_cov_omit:
            cmd += ["--python-cov-omit", omit]
    if enable_device_oracle:
        cmd += [
            "--enable-device-oracle",
            "--device-oracle-rtol",
            str(device_oracle_rtol),
            "--device-oracle-atol",
            str(device_oracle_atol),
        ]
        for dev in device_oracle_devices:
            cmd += ["--device-oracle-device", str(dev)]
        if edge_oracle_mutations:
            cmd += ["--edge-oracle-mutations"]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timeout = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timeout = True
        proc.kill()
        stdout, stderr = proc.communicate()
    duration_ms = round((time.perf_counter() - start) * 1000.0, 4)
    payload: Dict[str, Any] = {
        "exit_code": int(proc.returncode if proc.returncode is not None else -1),
        "timeout": timeout,
        "stdout": stdout[-4000:] if stdout else "",
        "stderr": stderr[-4000:] if stderr else "",
        "duration_ms": duration_ms,
        "output_json": output_json,
        "bundles": [],
        "python_coverage": {"enabled": False},
    }
    if os.path.exists(output_json):
        try:
            loaded = load_json(output_json)
            payload["bundles"] = loaded.get("bundles", []) if isinstance(loaded.get("bundles"), list) else []
            payload["python_coverage"] = loaded.get("python_coverage", {"enabled": False})
            payload["worker_pid"] = loaded.get("worker_pid", "")
        except Exception as exc:
            payload["load_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def append_bundle_outputs(
    bundle: Dict[str, Any],
    results_json_path: str,
    results_rows: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    worker_exit_code: int,
    worker_timeout: bool,
) -> None:
    bundle["worker_exit_code"] = worker_exit_code
    bundle["worker_timeout"] = worker_timeout
    metrics = compute_bundle_metrics(bundle)
    bundle["valid_testcase_hashes"] = metrics.get("valid_testcase_hashes", [])
    write_json(results_json_path, bundle)
    base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
    summary["executed_apis"] += 1
    if bundle.get("materialization_errors"):
        summary["materialization_fail"] += 1
    if metrics["base_success"]:
        summary["base_success"] += 1
    else:
        summary["base_fail"] += 1
    summary["mutation_cases"] += int(metrics["mutation_count"])
    summary["mutation_success"] += int(metrics["mutation_success"])
    summary["mutation_fail"] += int(metrics["mutation_fail"])
    summary["negative_expected_fail"] += int(metrics["negative_expected_fail"])
    summary["expected_negative_rejection"] += int(metrics["expected_negative_rejection"])
    summary["negative_accepted_not_bug"] += int(metrics["negative_accepted_not_bug"])
    summary["invalid_valid_mutation"] += int(metrics["invalid_valid_mutation"])
    summary["device_oracle_mismatch"] = int(summary.get("device_oracle_mismatch", 0) or 0) + int(metrics["device_oracle_mismatch"])
    summary["valid_programs"] = int(summary.get("valid_programs", 0) or 0) + int(metrics["valid_programs"])
    summary["total_valid_programs"] = int(summary.get("total_valid_programs", 0) or 0) + int(metrics["valid_programs"])
    summary.setdefault("_valid_program_hashes", [])
    if isinstance(summary.get("_valid_program_hashes"), list):
        summary["_valid_program_hashes"].extend(metrics.get("valid_testcase_hashes", []))
    summary["unique_valid_programs"] = len(set(str(x) for x in summary.get("_valid_program_hashes", []) if str(x)))
    if metrics["nan_observed"]:
        summary["apis_with_nan"] += 1

    bundle_has_failure = False
    if bundle.get("materialization_errors"):
        bundle_has_failure = True
        failure_rows.append({
            "api_full_name": bundle.get("api_full_name", ""),
            "stage": "materialize",
            "mutation_intent": "",
            "param": "",
            "rule": "",
            "error_type": "materialization_error",
            "error": " | ".join(bundle.get("materialization_errors", [])),
            "worker_exit_code": worker_exit_code,
            "worker_timeout": worker_timeout,
            "results_json_path": results_json_path,
        })
    if base and not metrics["base_success"]:
        bundle_has_failure = True
        failure_rows.append({
            "api_full_name": bundle.get("api_full_name", ""),
            "stage": "base_execution",
            "mutation_intent": "",
            "param": "",
            "rule": "",
            "error_type": base.get("error_type", ""),
            "error": base.get("error", ""),
            "worker_exit_code": worker_exit_code,
            "worker_timeout": worker_timeout,
            "results_json_path": results_json_path,
        })
    for mutation in bundle.get("mutations", []):
        if not isinstance(mutation, dict) or mutation.get("success") or mutation.get("expected_failure"):
            continue
        classification = str(mutation.get("classification", "") or "")
        bundle_has_failure = True
        failure_rows.append({
            "api_full_name": bundle.get("api_full_name", ""),
            "stage": "mutation_generator" if classification == "invalid_valid_mutation" else "mutation",
            "mutation_intent": mutation.get("mutation_intent", ""),
            "param": mutation.get("param", ""),
            "rule": mutation.get("rule", ""),
            "error_type": classification or mutation.get("error_type", ""),
            "error": mutation.get("error", ""),
            "worker_exit_code": worker_exit_code,
            "worker_timeout": worker_timeout,
            "results_json_path": results_json_path,
        })
    if bundle_has_failure:
        summary["apis_with_failures"] += 1

    results_rows.append({
        "api_full_name": bundle.get("api_full_name", ""),
        "import_path": bundle.get("import_path", ""),
        "backend": bundle.get("backend", ""),
        "library_version": bundle.get("library_version", ""),
        "init_json_path": bundle.get("init_json_path", ""),
        "seed": bundle.get("seed", ""),
        "base_success": str(metrics["base_success"]),
        "base_error": base.get("error", ""),
        "mutation_cases": metrics["mutation_count"],
        "mutation_success": metrics["mutation_success"],
        "mutation_fail": metrics["mutation_fail"],
        "negative_expected_fail": metrics["negative_expected_fail"],
        "expected_negative_rejection": metrics["expected_negative_rejection"],
        "negative_accepted_not_bug": metrics["negative_accepted_not_bug"],
        "invalid_valid_mutation": metrics["invalid_valid_mutation"],
        "device_oracle_mismatch": metrics["device_oracle_mismatch"],
        "valid_programs": metrics["valid_programs"],
        "nan_observed": metrics["nan_observed"],
        "worker_exit_code": worker_exit_code,
        "worker_timeout": worker_timeout,
        "results_json_path": results_json_path,
    })


def make_coverage_row(
    api_full_name: str,
    scope: str,
    py_summary: Dict[str, Any],
    native_summary: Dict[str, Any],
    init_json_path: str = "",
    results_json_path: str = "",
    worker: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    worker = worker or {}
    return {
        "api_full_name": api_full_name,
        "coverage_scope": scope,
        "python_statement_percent": py_summary.get("statement_percent", py_summary.get("covered_percent", "")),
        "python_branch_percent": py_summary.get("branch_percent", ""),
        "python_covered_lines": py_summary.get("covered_lines", ""),
        "python_coverable_lines": py_summary.get("num_statements", ""),
        "native_line_percent": native_summary.get("line_percent", ""),
        "native_covered_lines": native_summary.get("line_covered", ""),
        "native_coverable_lines": native_summary.get("line_total", ""),
        "native_branch_percent": native_summary.get("branch_percent", ""),
        "native_function_percent": native_summary.get("function_percent", ""),
        "python_json_path": py_summary.get("json_path", ""),
        "native_json_path": native_summary.get("summary_json", ""),
        "python_export_error": py_summary.get("export_error", py_summary.get("parse_error", py_summary.get("html_error", ""))),
        "native_export_error": native_summary.get("export_error", native_summary.get("parse_error", native_summary.get("html_error", ""))),
        "worker_exit_code": worker.get("exit_code", ""),
        "worker_timeout": worker.get("timeout", ""),
        "init_json_path": init_json_path,
        "results_json_path": results_json_path,
    }


def export_combined_python_coverage(
    per_api_python_dir: str,
    campaign_python_dir: str,
    python_cov_source: Sequence[str],
    python_cov_omit: Sequence[str],
) -> Dict[str, Any]:
    if Coverage is None:
        return {"enabled": False, "export_error": "coverage.py is not installed"}
    source_dir = Path(per_api_python_dir)
    data_files = [p for p in source_dir.rglob(".coverage*") if p.is_file()]
    if not data_files:
        return {"enabled": False, "export_error": "no coverage.py data files were produced by Stage 4 workers"}
    os.makedirs(campaign_python_dir, exist_ok=True)
    data_file = os.path.join(campaign_python_dir, ".coverage")
    json_path = os.path.join(campaign_python_dir, "coverage.json")
    html_dir = os.path.join(campaign_python_dir, "html")
    summary: Dict[str, Any] = {"enabled": True, "data_file": data_file, "combined_data_files": len(data_files)}
    try:
        cov = Coverage(
            data_file=data_file,
            branch=True,
            source=list(python_cov_source or []) or None,
            omit=list(python_cov_omit or []) or None,
        )
        cov.erase()
        cov.combine(data_paths=[str(path) for path in data_files])
        cov.save()
        cov.json_report(outfile=json_path, pretty_print=True)
        summary["json_path"] = json_path
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            totals = data.get("totals", {}) if isinstance(data, dict) else {}
            if isinstance(totals, dict):
                summary["statement_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["covered_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["covered_lines"] = totals.get("covered_lines", "")
                summary["num_statements"] = totals.get("num_statements", "")
                nb = totals.get("num_branches", 0) or 0
                cb = totals.get("covered_branches", 0) or 0
                if nb:
                    summary["branch_percent"] = round(100.0 * float(cb) / float(nb), 4)
        except Exception as exc:
            summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        try:
            cov.html_report(directory=html_dir)
            summary["html_dir"] = html_dir
        except Exception as exc:
            summary["html_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        summary["enabled"] = False
        summary["export_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def finalize_valid_program_counts(summary: Dict[str, Any]) -> None:
    hashes = [str(x) for x in summary.get("_valid_program_hashes", []) if str(x)]
    total = int(summary.get("total_valid_programs", summary.get("valid_programs", 0)) or 0)
    summary["valid_programs"] = total
    summary["total_valid_programs"] = total
    summary["unique_valid_programs"] = len(set(hashes)) if hashes else total
    summary.pop("_valid_program_hashes", None)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def excerpt(text: Any, limit: int = 2000) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


def stable_bug_signature(api: str, category: str, error_type: str, message: str, exit_code: Any) -> str:
    raw = "|".join([str(api), str(category), str(error_type), str(message)[:1000], str(exit_code)])
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def signal_name(exit_code: Any) -> str:
    try:
        code = int(exit_code)
    except Exception:
        return ""
    if code >= 0:
        return ""
    sig = -code
    try:
        return signal.Signals(sig).name
    except Exception:
        return f"SIG{sig}"


def classify_bug_category(error_type: str = "", message: str = "", exit_code: Any = 0, timeout: bool = False, nan_or_inf: bool = False) -> str:
    low = f"{error_type} {message}".lower()
    sig = signal_name(exit_code)
    if error_type == "DifferentialMismatch" or "differential mismatch" in low or "device outputs differ" in low:
        return "differential_mismatch"
    if timeout:
        return "timeout_or_hang"
    if sig in {"SIGSEGV", "SIGBUS"} or "segmentation fault" in low or "segfault" in low:
        return "segfault"
    try:
        code = int(exit_code)
    except Exception:
        code = 0
    if code not in {0, None} and (sig or "abort" in low or "core dumped" in low or "fatal" in low or "check failed" in low or "f0000" in low):
        return "crash_or_abort"
    if nan_or_inf:
        return "nan_or_inf_output"
    if error_type in {"AssertionError"} or "assert" in low:
        return "assertion_failure"
    if error_type in {"MemoryError", "ResourceExhaustedError"} or any(x in low for x in ["out of memory", "resource exhausted", "cannot allocate"]):
        return "resource_error"
    if error_type:
        return "unexpected_exception_on_valid_input"
    return "unknown"


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

PIPELINE_GENERATED_ERROR_TYPES = {
    "MaterializationError",
    "materialization_error",
}


def is_pipeline_generated_event(event: Dict[str, Any]) -> bool:
    error_type = str(event.get("error_type", "") or "")
    message = str(event.get("error", "") or "")
    classification = str(event.get("classification", "") or "")
    if error_type in PIPELINE_GENERATED_ERROR_TYPES:
        return True
    if classification in {"invalid_valid_mutation", "seed_materialization_failure", "pipeline_failure"}:
        return True
    return "was not materialized in base call" in message or "materialization failed" in message


def target_library_from_rows(rows: Sequence[Dict[str, Any]]) -> str:
    for row in rows:
        backend = str(row.get("backend", "") or "").strip()
        if backend and backend != "python":
            return backend
    for row in rows:
        api = str(row.get("api_full_name", "") or "")
        if "." in api:
            return api.split(".", 1)[0]
        if api:
            return api
    return "unknown"


def target_version_from_rows(rows: Sequence[Dict[str, Any]]) -> str:
    for row in rows:
        version = str(row.get("library_version", "") or "").strip()
        if version:
            return version
    return collect_target_version(target_library_from_rows(rows))


def collect_target_version(target_library: str) -> str:
    lib = str(target_library or "").strip()
    if not lib or lib == "unknown":
        return ""
    try:
        module = importlib.import_module(lib)
        return str(getattr(module, "__version__", "") or "").strip()
    except Exception:
        return ""


def _available(value: Any) -> Dict[str, Any]:
    return {"available": True, "value": value}


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": str(reason or "unavailable")}


def collect_environment_metadata(
    target_library: str,
    target_version: str,
    seed: Any,
    mutation_budget: Any,
    run_id: str,
    coverage_method: str,
) -> Dict[str, Any]:
    lib = str(target_library or "unknown")
    metadata: Dict[str, Any] = {
        "target_library": lib,
        "target_version": target_version or collect_target_version(lib),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "mutation_budget": mutation_budget,
        "run_id": run_id or "",
        "coverage_method": coverage_method,
    }
    try:
        module = importlib.import_module(lib)
        metadata["target_module_version"] = _available(str(getattr(module, "__version__", "") or ""))
    except Exception as exc:
        metadata["target_module_version"] = _unavailable(f"{type(exc).__name__}: {exc}")

    if lib == "jax":
        try:
            import jax  # type: ignore

            metadata["jax_version"] = _available(str(getattr(jax, "__version__", "") or ""))
            try:
                metadata["jax_backend"] = _available(str(jax.default_backend()))
            except Exception as exc:
                metadata["jax_backend"] = _unavailable(f"{type(exc).__name__}: {exc}")
            try:
                metadata["jax_devices"] = _available([str(device) for device in jax.devices()])
            except Exception as exc:
                metadata["jax_devices"] = _unavailable(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            metadata["jax_version"] = _unavailable(reason)
            metadata["jax_backend"] = _unavailable(reason)
            metadata["jax_devices"] = _unavailable(reason)
        try:
            import jaxlib  # type: ignore

            metadata["jaxlib_version"] = _available(str(getattr(jaxlib, "__version__", "") or ""))
        except Exception as exc:
            metadata["jaxlib_version"] = _unavailable(f"{type(exc).__name__}: {exc}")
    else:
        metadata["jax_version"] = _unavailable("target library is not jax")
        metadata["jaxlib_version"] = _unavailable("target library is not jax")
        metadata["jax_backend"] = _unavailable("target library is not jax")
        metadata["jax_devices"] = _unavailable("target library is not jax")

    return metadata


def reproduce_command(init_path: str, results_dir: str, mutation_budget: int, seed: Any) -> str:
    out_path = os.path.join(results_dir, "repro", f"repro_{safe_name(Path(init_path).stem)}.json")
    return (
        "python3 -m stage4.stage4_worker "
        f"--init-path {shlex.quote(init_path)} "
        f"--results-json {shlex.quote(out_path)} "
        f"--mutation-budget {int(mutation_budget)} --seed {seed}"
    )


def bug_from_event(
    api: str,
    target_library: str,
    target_version: str,
    init_path: str,
    results_path: str,
    stage: str,
    event: Dict[str, Any],
    worker_exit_code: Any,
    worker_timeout: bool,
    mutation_budget: int,
    seed: Any,
    stdout_stderr: str = "",
) -> Optional[Dict[str, Any]]:
    if is_pipeline_generated_event(event):
        return None
    error_type = str(event.get("error_type", "") or "")
    message = str(event.get("error", "") or "")
    if event.get("differential_mismatch"):
        error_type = "DifferentialMismatch"
        oracle = event.get("device_oracle", {}) if isinstance(event.get("device_oracle"), dict) else {}
        message = str(oracle.get("reason", "") or message or "device outputs differ")
    nan_or_inf = bool(event.get("has_nan") or event.get("has_inf"))
    try:
        code = int(event.get("exit_code", worker_exit_code))
    except Exception:
        code = 0
    effective_timeout = bool(worker_timeout or event.get("timeout"))
    if not effective_timeout and code == 0 and error_type in NORMAL_VALIDATION_ERROR_TYPES and not nan_or_inf:
        return None
    category = classify_bug_category(error_type, message, code, effective_timeout, nan_or_inf)
    if category == "unknown":
        return None
    sig = stable_bug_signature(api, category, error_type, message, code)
    bug_id = f"bug-{sig}"
    return {
        "bug_id": bug_id,
        "api": api,
        "stage": stage,
        "category": category,
        "target_library": target_library,
        "target_version": target_version,
        "testcase": results_path,
        "input": init_path,
        "command_to_reproduce": reproduce_command(init_path, str(Path(results_path).parents[1]), mutation_budget, seed),
        "stdout_stderr_excerpt": excerpt(stdout_stderr or message),
        "traceback": excerpt(event.get("traceback", "") or message),
        "signature": sig,
        "exit_code": code,
        "signal": signal_name(code),
        "first_seen_at": utc_now(),
        "duplicate_of": None,
        "status": "candidate",
        "oracle_verdict": "candidate_implementation_bug",
        "device_oracle": event.get("device_oracle", {}) if isinstance(event.get("device_oracle"), dict) else {},
    }


def _enum_values_for_event(init_obj: Dict[str, Any], api: str, param: str) -> List[str]:
    params = init_obj.get("params", {}) if isinstance(init_obj.get("params"), dict) else {}
    meta = params.get(param, {}) if isinstance(params.get(param), dict) else {}
    spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else {}
    values: List[str] = []
    for key in ("enum_values", "valid_values", "enum"):
        raw = spec.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            values.extend(str(x) for x in raw if str(x))
    if api.startswith("jax.") and param.lower() == "padding":
        values.extend(["VALID", "SAME", "SAME_LOWER"])
    return list(dict.fromkeys(values))


def expected_invalid_input_category(
    api: str,
    init_obj: Dict[str, Any],
    event: Dict[str, Any],
    worker_exit_code: Any = 0,
    worker_timeout: bool = False,
) -> Tuple[str, str]:
    if event.get("success") or worker_timeout:
        return "", ""
    try:
        code = int(event.get("exit_code", worker_exit_code))
    except Exception:
        code = 0
    if code != 0:
        return "", ""
    error_type = str(event.get("error_type", "") or "")
    classification = str(event.get("classification", "") or "")
    intent = str(event.get("mutation_intent", "") or "")
    if classification == "expected_negative_rejection":
        return "expected_exception_on_invalid_input", "negative mutation was rejected by normal input validation"
    if classification == "invalid_valid_mutation":
        return "invalid_generated_mutation", "intended-valid mutation violated schema constraints"
    if intent == "negative" and error_type in NORMAL_VALIDATION_ERROR_TYPES:
        return "expected_exception_on_invalid_input", "known-invalid negative input raised a normal validation exception"
    if intent == "valid" and error_type in NORMAL_VALIDATION_ERROR_TYPES:
        param = str(event.get("param", "") or "")
        rule = str(event.get("rule", "") or "")
        message = str(event.get("error", "") or "")
        enum_values = _enum_values_for_event(init_obj, api, param)
        if enum_values and (rule in {"string_replace", "enum_invalid"} or "_mut" in message):
            return "invalid_generated_mutation", "intended-valid enum mutation generated an out-of-domain value"
    return "", ""


def _bug_event_candidates(bundle: Dict[str, Any], bug: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage = str(bug.get("stage", "") or "")
    if stage == "base_execution":
        base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
        return [base] if base else []
    if stage in {"mutation", "stage4_worker", ""}:
        muts = bundle.get("mutations", []) if isinstance(bundle.get("mutations"), list) else []
        return [m for m in muts if isinstance(m, dict)]
    return []


def _event_matches_bug(api: str, bug: Dict[str, Any], event: Dict[str, Any], worker_exit_code: Any) -> bool:
    error_type = str(event.get("error_type", "") or "")
    message = str(event.get("error", "") or "")
    try:
        code = int(event.get("exit_code", worker_exit_code))
    except Exception:
        code = 0
    bug_sig = str(bug.get("signature", "") or "")
    categories = [
        str(bug.get("category", "") or ""),
        "unexpected_exception_on_valid_input",
        "invalid_generated_mutation",
        "expected_exception_on_invalid_input",
    ]
    for category in dict.fromkeys(c for c in categories if c):
        if stable_bug_signature(api, category, error_type, message, code) == bug_sig:
            return True
    evidence = " ".join(str(bug.get(key, "") or "") for key in ("stdout_stderr_excerpt", "traceback", "reclassification_reason"))
    return bool(message and (message in evidence or evidence in message))


def refresh_bug_summary(bug_report: Dict[str, Any]) -> None:
    bugs = [b for b in bug_report.get("bugs", []) if isinstance(b, dict)]
    unique = [b for b in bugs if not b.get("duplicate_of")]
    summary = bug_report.get("summary") if isinstance(bug_report.get("summary"), dict) else {}
    summary["total_candidates"] = len([
        b
        for b in unique
        if b.get("status") not in {"unresolved", "false_positive"}
        and b.get("category") not in {"invalid_generated_mutation", "expected_exception_on_invalid_input"}
    ])
    summary["reproduced"] = sum(1 for b in unique if b.get("status") == "reproduced")
    summary["known_existing_caught"] = sum(1 for b in unique if b.get("status") == "known_existing")
    summary["false_positives"] = sum(1 for b in unique if b.get("status") == "false_positive")
    summary["unresolved"] = sum(1 for b in unique if b.get("status") == "unresolved")
    summary["valid_candidate_bugs"] = summary["total_candidates"]
    bug_report["summary"] = summary
    bug_report["candidate_bugs"] = [
        b
        for b in unique
        if b.get("status") not in {"unresolved", "false_positive"}
        and b.get("category") not in {"invalid_generated_mutation", "expected_exception_on_invalid_input", "unresolved_pipeline_failure"}
    ]


def reclassify_bug_report(bug_report: Dict[str, Any]) -> Dict[str, Any]:
    bugs = bug_report.get("bugs", [])
    if not isinstance(bugs, list):
        bug_report["bugs"] = []
        refresh_bug_summary(bug_report)
        return bug_report
    for bug in bugs:
        if not isinstance(bug, dict) or bug.get("status") in {"unresolved", "false_positive"}:
            continue
        results_path = str(bug.get("testcase") or bug.get("reproducer_path") or "")
        if not results_path:
            continue
        path = Path(results_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        try:
            bundle = load_json(str(path))
        except Exception:
            continue
        init_path = str(bundle.get("init_json_path") or bug.get("input") or "")
        init_obj: Dict[str, Any] = {}
        if init_path:
            ipath = Path(init_path)
            if not ipath.is_absolute():
                ipath = ROOT / ipath
            if ipath.exists():
                try:
                    init_obj = load_json(str(ipath))
                except Exception:
                    init_obj = {}
        api = str(bundle.get("api_full_name") or bug.get("api") or "")
        worker_exit_code = bundle.get("worker_exit_code", bug.get("exit_code", 0))
        worker_timeout = bool(bundle.get("worker_timeout", False))
        candidates = _bug_event_candidates(bundle, bug)
        matched = [event for event in candidates if _event_matches_bug(api, bug, event, worker_exit_code)]
        if not matched and len(candidates) == 1:
            matched = candidates
        for event in matched:
            category, reason = expected_invalid_input_category(api, init_obj, event, worker_exit_code, worker_timeout)
            if not category:
                continue
            bug["original_category"] = bug.get("category", "")
            bug["category"] = category
            bug["status"] = "false_positive"
            bug["oracle_decision"] = "reclassified_false_positive"
            bug["reclassification_reason"] = reason
            break
    refresh_bug_summary(bug_report)
    return bug_report


def load_known_bug_signatures(path: str) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    rows = data.get("bugs", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return set()
    out: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        sig = str(row.get("signature", "") or "").strip()
        if sig:
            out.add(sig)
    return out


def load_unresolved_failure_bugs(path: str, target_library: str, target_version: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    bugs: List[Dict[str, Any]] = []
    for row in read_csv_rows(path):
        api = str(row.get("api") or row.get("api_full_name") or "").strip()
        if not api:
            continue
        stage = str(row.get("stage") or row.get("root_stage") or "").strip() or "repair_oracle"
        message = str(row.get("message") or row.get("error") or row.get("reason") or row.get("errors") or "").strip()
        error_type = str(row.get("error_type") or row.get("status") or row.get("classification") or "unresolved").strip()
        signature = stable_bug_signature(api, "unresolved_pipeline_failure", error_type, message, 0)
        bugs.append({
            "bug_id": f"unresolved-{signature}",
            "api": api,
            "stage": stage,
            "category": "unresolved_pipeline_failure",
            "target_library": target_library,
            "target_version": target_version,
            "testcase": str(row.get("payload_ref") or row.get("results_json_path") or row.get("init_json_path") or ""),
            "input": str(row.get("init_json_path") or ""),
            "command_to_reproduce": "",
            "stdout_stderr_excerpt": excerpt(message),
            "traceback": excerpt(str(row.get("traceback") or message)),
            "signature": signature,
            "exit_code": "",
            "signal": "",
            "first_seen_at": utc_now(),
            "duplicate_of": None,
            "status": "unresolved",
            "oracle_decision": str(row.get("oracle_decision") or "insufficient_grounded_evidence"),
        })
    return bugs


def build_bug_report(
    results_dir: str,
    results_rows: Sequence[Dict[str, Any]],
    mutation_budget: int,
    known_bugs_file: str = "",
    unresolved_failures_file: str = "",
) -> Dict[str, Any]:
    target_library = target_library_from_rows(results_rows)
    target_version = target_version_from_rows(results_rows)
    bugs: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    treat_nan_as_bug = os.environ.get("DEEPFUZZ_TREAT_VALID_NAN_AS_BUG", "").strip().lower() in {"1", "true", "yes"}
    report_base_failures_as_bugs = os.environ.get("DEEPFUZZ_REPORT_BASE_FAILURES_AS_BUGS", "").strip().lower() in {"1", "true", "yes"}

    for row in results_rows:
        results_path = str(row.get("results_json_path", "") or "")
        if not results_path or not os.path.exists(results_path):
            continue
        try:
            bundle = load_json(results_path)
        except Exception:
            continue
        api = str(bundle.get("api_full_name", "") or row.get("api_full_name", "") or "")
        init_path = str(bundle.get("init_json_path", "") or row.get("init_json_path", "") or "")
        seed = bundle.get("seed", row.get("seed", ""))
        worker_exit_code = bundle.get("worker_exit_code", row.get("worker_exit_code", 0))
        worker_timeout = bool(bundle.get("worker_timeout", row.get("worker_timeout", False)))
        stdout_stderr = "\n".join(
            x for x in [str(bundle.get("worker_stdout", "") or ""), str(bundle.get("worker_stderr", "") or "")] if x
        )

        base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
        base_success = bool(base.get("success"))
        if base:
            if base.get("differential_mismatch"):
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "base_execution", base, worker_exit_code, False, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)
            if treat_nan_as_bug and base.get("success") and (base.get("has_nan") or base.get("has_inf")):
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "base_execution", base, 0, False, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)
            elif report_base_failures_as_bugs and bundle.get("base_kwargs_materialized") and not base.get("success") and base.get("invoked_api", True):
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "base_execution", base, worker_exit_code, worker_timeout, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)

        for mutation in bundle.get("mutations", []):
            if not isinstance(mutation, dict):
                continue
            intent = str(mutation.get("mutation_intent", "valid") or "valid")
            if mutation.get("expected_failure") or is_pipeline_generated_event(mutation):
                continue
            if intent == "valid" and mutation.get("differential_mismatch"):
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "mutation", mutation, worker_exit_code, False, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)
            elif treat_nan_as_bug and mutation.get("success") and (mutation.get("has_nan") or mutation.get("has_inf")) and intent == "valid":
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "mutation", mutation, 0, False, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)
            elif intent == "valid" and not mutation.get("success") and not mutation.get("expected_failure"):
                bug = bug_from_event(api, target_library, target_version, init_path, results_path, "mutation", mutation, worker_exit_code, worker_timeout, mutation_budget, seed, stdout_stderr)
                if bug:
                    bugs.append(bug)

    excluded_pipeline_issues = load_unresolved_failure_bugs(unresolved_failures_file, target_library, target_version)

    for bug in bugs:
        sig = str(bug.get("signature", ""))
        if sig in seen:
            bug["duplicate_of"] = seen[sig]
        else:
            seen[sig] = str(bug.get("bug_id", ""))

    known = load_known_bug_signatures(known_bugs_file)
    if known:
        for bug in bugs:
            if str(bug.get("signature", "")) in known:
                bug["status"] = "known_existing"

    report = {
        "schema_version": "2.0",
        "target_library": target_library,
        "target_version": target_version,
        "bugs": bugs,
        "candidate_bugs": bugs,
        "excluded_pipeline_issues": {
            "count": len([b for b in excluded_pipeline_issues if not b.get("duplicate_of")]),
            "summary": "Pipeline/config/coverage/seed failures are excluded from candidate library bugs.",
            "issues": excluded_pipeline_issues[:50],
        },
        "summary": {
            "total_candidates": 0,
            "reproduced": sum(1 for b in bugs if b.get("status") == "reproduced" and not b.get("duplicate_of")),
            "known_existing_caught": sum(1 for b in bugs if b.get("status") == "known_existing" and not b.get("duplicate_of")),
            "false_positives": sum(1 for b in bugs if b.get("status") == "false_positive" and not b.get("duplicate_of")),
            "unresolved": 0,
            "excluded_pipeline_issues": len([b for b in excluded_pipeline_issues if not b.get("duplicate_of")]),
        },
    }
    return reclassify_bug_report(report)


BUG_CSV_FIELDS = [
    "bug_id",
    "api",
    "category",
    "oracle_verdict",
    "stage",
    "status",
    "target_library",
    "target_version",
    "reproducer_path",
    "reproduce_command",
    "error_signature",
    "exit_code",
    "first_seen_at",
    "duplicate_of",
]


def _bug_csv_rows(bugs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        rows.append({
            "bug_id": bug.get("bug_id", ""),
            "api": bug.get("api", ""),
            "category": bug.get("category", ""),
            "oracle_verdict": bug.get("oracle_verdict", "candidate_implementation_bug" if bug.get("status") == "candidate" else ""),
            "stage": bug.get("stage", ""),
            "status": bug.get("status", ""),
            "target_library": bug.get("target_library", ""),
            "target_version": bug.get("target_version", ""),
            "reproducer_path": bug.get("testcase", bug.get("input", "")),
            "reproduce_command": bug.get("command_to_reproduce", ""),
            "error_signature": bug.get("signature", ""),
            "exit_code": bug.get("exit_code", ""),
            "first_seen_at": bug.get("first_seen_at", ""),
            "duplicate_of": bug.get("duplicate_of", ""),
        })
    return rows


def write_bug_report_csv(path: str | Path, bug_report: Dict[str, Any]) -> None:
    candidates = bug_report.get("candidate_bugs", [])
    if not isinstance(candidates, list):
        candidates = []
    write_csv(path, _bug_csv_rows(candidates), BUG_CSV_FIELDS)


def write_bug_audit_csv(path: str | Path, bug_report: Dict[str, Any]) -> None:
    bugs = bug_report.get("bugs", [])
    if not isinstance(bugs, list):
        bugs = []
    rows = _bug_csv_rows(bugs)
    write_csv(path, rows, BUG_CSV_FIELDS)


EXECUTION_SUMMARY_FIELDS = [
    "api_full_name",
    "case_kind",
    "case_index",
    "mutation_intent",
    "param",
    "rule",
    "success",
    "classification",
    "error_type",
    "error",
    "results_json_path",
]

FAILURE_AUDIT_FIELDS = [
    "api_full_name",
    "stage",
    "case_kind",
    "mutation_intent",
    "param",
    "rule",
    "classification",
    "error_type",
    "error",
    "results_json_path",
]


def _audit_classification_for_base(bundle: Dict[str, Any], base: Dict[str, Any]) -> str:
    if base.get("success"):
        return "success_valid"
    if bundle.get("materialization_errors") or not bundle.get("base_kwargs_materialized", True) or not base.get("invoked_api", True):
        return "seed_materialization_failure"
    return "pipeline_failure"


def build_execution_and_audit_rows(results_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    execution_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for row in results_rows:
        results_path = str(row.get("results_json_path", "") or "")
        if not results_path or not os.path.exists(results_path):
            continue
        try:
            bundle = load_json(results_path)
        except Exception:
            continue
        api = str(bundle.get("api_full_name", "") or row.get("api_full_name", "") or "")
        base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
        base_classification = "differential_mismatch" if base.get("differential_mismatch") else _audit_classification_for_base(bundle, base)
        execution_rows.append({
            "api_full_name": api,
            "case_kind": "base_valid",
            "case_index": 0,
            "mutation_intent": "valid",
            "param": "",
            "rule": "",
            "success": bool(base.get("success")),
            "classification": base_classification,
            "error_type": base.get("error_type", ""),
            "error": base.get("error", ""),
            "results_json_path": results_path,
        })
        if base_classification in {"seed_materialization_failure", "pipeline_failure", "differential_mismatch"}:
            audit_rows.append({
                "api_full_name": api,
                "stage": "materialize" if base_classification == "seed_materialization_failure" else "base_execution",
                "case_kind": "base_valid",
                "mutation_intent": "valid",
                "param": "",
                "rule": "",
                "classification": base_classification,
                "error_type": base.get("error_type", base_classification),
                "error": " | ".join(bundle.get("materialization_errors", []) or []) or base.get("error", ""),
                "results_json_path": results_path,
            })
        for idx, mutation in enumerate(bundle.get("mutations", []) if isinstance(bundle.get("mutations"), list) else [], start=1):
            if not isinstance(mutation, dict):
                continue
            intent = str(mutation.get("mutation_intent", "valid") or "valid")
            case_kind = "negative_mutation" if intent == "negative" else "valid_mutation"
            classification = str(mutation.get("classification", "") or "")
            if classification == "negative_mutation_accepted":
                classification = "negative_accepted_not_bug"
            execution_rows.append({
                "api_full_name": api,
                "case_kind": case_kind,
                "case_index": idx,
                "mutation_intent": intent,
                "param": mutation.get("param", ""),
                "rule": mutation.get("rule", ""),
                "success": bool(mutation.get("success")),
                "classification": classification,
                "error_type": mutation.get("error_type", ""),
                "error": mutation.get("error", ""),
                "results_json_path": results_path,
            })
            if classification in {"expected_negative_rejection", "negative_accepted_not_bug", "invalid_valid_mutation", "differential_mismatch"}:
                audit_rows.append({
                    "api_full_name": api,
                    "stage": case_kind,
                    "case_kind": case_kind,
                    "mutation_intent": intent,
                    "param": mutation.get("param", ""),
                    "rule": mutation.get("rule", ""),
                    "classification": classification,
                    "error_type": mutation.get("error_type", classification),
                    "error": mutation.get("error", ""),
                    "results_json_path": results_path,
                })
    return execution_rows, audit_rows


def _numeric(value: Any) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def _best_campaign_row(coverage_rows: Sequence[Dict[str, Any]], prefix: str) -> Optional[Dict[str, Any]]:
    for row in coverage_rows:
        if row.get("api_full_name") == "__CAMPAIGN__" and _numeric(row.get(f"{prefix}_line_percent" if prefix == "native" else "python_statement_percent")) is not None:
            return row
    return None


def _resolve_api_object(api_full_name: str) -> Any:
    parts = str(api_full_name or "").split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_name)
        except Exception:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return inspect.unwrap(obj)
        except Exception:
            return None
    return None


def _coverage_file_entry(files: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    if source_file in files and isinstance(files[source_file], dict):
        return files[source_file]
    abs_source = os.path.abspath(source_file)
    for key, value in files.items():
        if not isinstance(value, dict):
            continue
        try:
            if os.path.abspath(str(key)) == abs_source:
                return value
        except Exception:
            pass
    source_name = os.path.basename(source_file)
    matches = [
        value
        for key, value in files.items()
        if isinstance(value, dict) and os.path.basename(str(key)) == source_name
    ]
    return matches[0] if len(matches) == 1 else {}


def _load_coverage_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _selected_function_item(api: str, files: Dict[str, Any], reason_prefix: str = "") -> Dict[str, Any]:
    obj = _resolve_api_object(api)
    if obj is None:
        return {
            "api": api,
            "available": False,
            "reason": f"{reason_prefix}no_python_source",
            "function_line_coverage_percent": None,
            "function_source_file": "",
            "function_missing_lines": [],
        }
    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        source_file = ""
        source_lines = []
        start_line = 0
    if not source_file or not source_lines or not start_line:
        return {
            "api": api,
            "available": False,
            "reason": f"{reason_prefix}no_python_source",
            "function_line_coverage_percent": None,
            "function_source_file": source_file or "",
            "function_missing_lines": [],
        }

    span = set(range(int(start_line), int(start_line) + len(source_lines)))
    entry = _coverage_file_entry(files, source_file)
    if not entry:
        return {
            "api": api,
            "available": False,
            "reason": f"{reason_prefix}no_coverage_data_for_source_file",
            "function_line_coverage_percent": None,
            "function_source_file": os.path.abspath(source_file),
            "function_start_line": int(start_line),
            "function_end_line": int(start_line) + len(source_lines) - 1,
            "function_missing_lines": [],
        }
    executed = {int(x) for x in entry.get("executed_lines", []) or [] if str(x).strip().lstrip("-").isdigit()}
    missing = {int(x) for x in entry.get("missing_lines", []) or [] if str(x).strip().lstrip("-").isdigit()}
    excluded = {int(x) for x in entry.get("excluded_lines", []) or [] if str(x).strip().lstrip("-").isdigit()}
    if missing:
        coverable = ((executed | missing) & span) - excluded
    else:
        coverable = span - excluded
    covered = executed & coverable
    missing_in_function = sorted(coverable - covered)
    percent = round((len(covered) / len(coverable)) * 100.0, 4) if coverable else None
    return {
        "api": api,
        "available": percent is not None,
        "reason": "" if percent is not None else f"{reason_prefix}no_coverable_python_lines",
        "function_line_coverage_percent": percent,
        "function_covered_lines": len(covered),
        "function_coverable_lines": len(coverable),
        "function_missing_lines": missing_in_function,
        "function_source_file": os.path.abspath(source_file),
        "function_start_line": int(start_line),
        "function_end_line": int(start_line) + len(source_lines) - 1,
    }


def _selected_function_summary(
    per_api: Dict[str, Dict[str, Any]],
    unavailable: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total_covered = 0
    total_coverable = 0
    for item in per_api.values():
        if item.get("function_line_coverage_percent") is None:
            continue
        total_covered += int(item.get("function_covered_lines", 0) or 0)
        total_coverable += int(item.get("function_coverable_lines", 0) or 0)

    available = total_coverable > 0
    out = {
        "available": available,
        "reason": "" if available else "no_selected_python_source",
        "covered_lines": total_covered,
        "coverable_lines": total_coverable,
        "percent": round((total_covered / total_coverable) * 100.0, 4) if total_coverable else None,
        "excluded_unavailable_apis": unavailable,
        "per_api": per_api,
    }
    if extra:
        out.update(extra)
    return out


def compute_selected_function_coverage(
    python_json_path: str,
    results_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    data = _load_coverage_json(python_json_path)
    files = data.get("files", {}) if isinstance(data.get("files"), dict) else {}
    per_api: Dict[str, Dict[str, Any]] = {}
    unavailable: List[Dict[str, Any]] = []

    for row in results_rows:
        api = str(row.get("api_full_name", "") or "")
        if not api:
            continue
        item = _selected_function_item(api, files)
        per_api[api] = item
        if not item.get("available"):
            unavailable.append({"api": api, "reason": item.get("reason", "")})

    reason = "python_coverage_unavailable" if not python_json_path else "no_selected_python_source"
    summary = _selected_function_summary(per_api, unavailable, {"python_json_path": python_json_path, "aggregation": "campaign_python_json"})
    if not summary.get("available"):
        summary["reason"] = reason
    return summary


def compute_selected_function_coverage_from_per_api_rows(
    coverage_rows: Sequence[Dict[str, Any]],
    results_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    per_api_rows = {
        str(row.get("api_full_name", "") or ""): row
        for row in coverage_rows
        if str(row.get("api_full_name", "") or "") and str(row.get("api_full_name", "") or "") != "__CAMPAIGN__"
    }
    per_api: Dict[str, Dict[str, Any]] = {}
    unavailable: List[Dict[str, Any]] = []
    loaded_files: Dict[str, Dict[str, Any]] = {}
    used_paths: Set[str] = set()

    for row in results_rows:
        api = str(row.get("api_full_name", "") or "")
        if not api:
            continue
        coverage_path = str(per_api_rows.get(api, {}).get("python_json_path", "") or "")
        if not coverage_path:
            item = _selected_function_item(api, {}, reason_prefix="missing_per_api_coverage:")
            per_api[api] = item
            unavailable.append({"api": api, "reason": item.get("reason", "")})
            continue
        if coverage_path not in loaded_files:
            data = _load_coverage_json(coverage_path)
            loaded_files[coverage_path] = data.get("files", {}) if isinstance(data.get("files"), dict) else {}
        files = loaded_files[coverage_path]
        used_paths.add(coverage_path)
        item = _selected_function_item(api, files)
        per_api[api] = item
        if not item.get("available"):
            unavailable.append({"api": api, "reason": item.get("reason", "")})

    summary = _selected_function_summary(
        per_api,
        unavailable,
        {
            "aggregation": "per_api_python_json",
            "python_json_path": "",
            "python_json_paths_count": len(used_paths),
        },
    )
    if not summary.get("available"):
        summary["reason"] = "per_api_python_coverage_unavailable"
    return summary


def build_coverage_report(
    summary: Dict[str, Any],
    coverage_rows: Sequence[Dict[str, Any]],
    results_rows: Sequence[Dict[str, Any]],
    bug_report: Dict[str, Any],
    execution_time_seconds: float,
    native_requested: bool,
    python_requested: bool,
    run_id: str = "",
) -> Dict[str, Any]:
    total_api_count = int(summary.get("selected_apis", 0) or 0)
    covered_api_count = int(summary.get("base_success", 0) or 0)
    api_coverage_percent = round((covered_api_count / total_api_count) * 100.0, 4) if total_api_count else 0.0
    target_library = target_library_from_rows(results_rows)
    target_version = target_version_from_rows(results_rows)
    bugs_by_api: Dict[str, int] = {}
    candidate_bugs = bug_report.get("candidate_bugs", [])
    if not isinstance(candidate_bugs, list):
        candidate_bugs = []
    for bug in candidate_bugs:
        if not isinstance(bug, dict):
            continue
        if bug.get("duplicate_of"):
            continue
        api = str(bug.get("api", "") or "")
        bugs_by_api[api] = bugs_by_api.get(api, 0) + 1

    native_campaign = _best_campaign_row(coverage_rows, "native")
    python_campaign = _best_campaign_row(coverage_rows, "python")
    limitations: List[str] = []
    coverage_method = "api_coverage_only"
    line_percent: Optional[float] = None
    covered_lines: Optional[int] = None
    coverable_lines: Optional[int] = None
    main_number = api_coverage_percent
    main_label = "API execution coverage %"
    native_line_percent: Optional[float] = None
    python_line_percent: Optional[float] = None
    selected_function: Dict[str, Any] = {"available": False, "reason": "python_coverage_unavailable", "per_api": {}}

    if native_campaign:
        coverage_method = "mixed" if python_requested and python_campaign else "native_gcov_lcov"
        native_line_percent = _numeric(native_campaign.get("native_line_percent"))
        line_percent = native_line_percent
        covered_lines = int(_numeric(native_campaign.get("native_covered_lines")) or 0)
        coverable_lines = int(_numeric(native_campaign.get("native_coverable_lines")) or 0)
        if line_percent is not None:
            main_number = round(line_percent, 4)
            main_label = "native line coverage %"
    elif python_campaign:
        coverage_method = "python_coverage"
        python_line_percent = _numeric(python_campaign.get("python_statement_percent"))
        line_percent = python_line_percent
        covered_lines = int(_numeric(python_campaign.get("python_covered_lines")) or 0)
        coverable_lines = int(_numeric(python_campaign.get("python_coverable_lines")) or 0)
        if python_line_percent is not None:
            main_number = round(python_line_percent, 4)
            main_label = "Python wrapper line coverage %"
        limitations.append("Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.")
        selected_function = compute_selected_function_coverage(str(python_campaign.get("python_json_path", "") or ""), results_rows)
        per_api_selected_function = compute_selected_function_coverage_from_per_api_rows(coverage_rows, results_rows)
        if per_api_selected_function.get("available"):
            selected_function = per_api_selected_function
            selected_function["campaign_python_json_path"] = str(python_campaign.get("python_json_path", "") or "")
            limitations.append(
                "Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset."
            )
        if selected_function.get("available") and selected_function.get("percent") is not None:
            line_percent = _numeric(selected_function.get("percent"))
            covered_lines = int(selected_function.get("covered_lines", 0) or 0)
            coverable_lines = int(selected_function.get("coverable_lines", 0) or 0)
            main_number = round(float(selected_function["percent"]), 4)
            main_label = "Selected function line coverage %"
        elif result_by_api_count := len(results_rows):
            unavailable_count = len(selected_function.get("excluded_unavailable_apis", []) or [])
            if unavailable_count:
                limitations.append(
                    f"Selected-function coverage was unavailable for {unavailable_count}/{result_by_api_count} APIs; unavailable APIs are excluded from the selected-function denominator and retained in API execution coverage."
                )
    if python_campaign and python_line_percent is None:
        python_line_percent = _numeric(python_campaign.get("python_statement_percent"))
    if native_campaign and native_line_percent is None:
        native_line_percent = _numeric(native_campaign.get("native_line_percent"))
    if line_percent is None:
        limitations.append("Source-line coverage unavailable in this configuration; main metric is API execution coverage over the accepted API subset.")
        limitations.append("Stage 4 isolates each API in its own subprocess so native aborts do not kill the campaign; line coverage is reported only when a supported union export is available.")

    if native_requested and not native_campaign:
        limitations.append("Native union coverage was requested but no usable GCOV/LCOV campaign summary was produced.")
    if not native_campaign:
        limitations.append("Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.")
    if total_api_count == 0:
        limitations.append("No Stage 4 APIs were selected.")

    per_api_rows = {str(r.get("api_full_name", "")): r for r in coverage_rows if r.get("api_full_name") != "__CAMPAIGN__"}
    result_by_api = {str(r.get("api_full_name", "")): r for r in results_rows}
    per_api: List[Dict[str, Any]] = []
    low_coverage_apis: List[Dict[str, Any]] = []
    low_threshold = _numeric(summary.get("low_coverage_threshold"))
    if low_threshold is None:
        low_threshold = 60.0
    for api in sorted(result_by_api):
        rr = result_by_api[api]
        cr = per_api_rows.get(api, {})
        native_line = _numeric(cr.get("native_line_percent"))
        python_line = _numeric(cr.get("python_statement_percent"))
        fn = selected_function.get("per_api", {}).get(api, {}) if isinstance(selected_function.get("per_api"), dict) else {}
        fn_percent = fn.get("function_line_coverage_percent")
        fn_percent_num = _numeric(fn_percent)
        if bool(fn.get("available", False)) and fn_percent_num is not None and fn_percent_num < low_threshold:
            low_coverage_apis.append({
                "api_full_name": api,
                "reason": f"selected_function_line_coverage_percent={fn_percent_num:g} < {low_threshold:g}",
                "function_line_coverage_percent": fn_percent_num,
                "function_covered_lines": fn.get("function_covered_lines"),
                "function_coverable_lines": fn.get("function_coverable_lines"),
                "function_source_file": fn.get("function_source_file", ""),
            })
        per_api.append({
            "api": api,
            "covered": str(rr.get("base_success", "")).lower() == "true",
            "valid_programs": int(rr.get("valid_programs", 0) or 0),
            "bugs": bugs_by_api.get(api, 0),
            "line_coverage_percent": None if coverage_method == "api_coverage_only" else (native_line if native_line is not None else python_line),
            "function_line_coverage_percent": fn_percent,
            "function_covered_lines": fn.get("function_covered_lines"),
            "function_coverable_lines": fn.get("function_coverable_lines"),
            "function_missing_lines": fn.get("function_missing_lines", []),
            "function_source_file": fn.get("function_source_file", ""),
            "function_coverage_available": bool(fn.get("available", False)),
            "function_coverage_reason": fn.get("reason", ""),
        })

    environment_metadata = collect_environment_metadata(
        target_library=target_library,
        target_version=target_version,
        seed=summary.get("seed", ""),
        mutation_budget=summary.get("mutation_budget", ""),
        run_id=run_id or str(summary.get("run_id", "") or ""),
        coverage_method=coverage_method,
    )

    selected_percent = _numeric(selected_function.get("percent"))
    low_count: Any = summary.get("apis_with_low_coverage", 0)
    low_applicability = "not_applicable_source_line_coverage_unavailable" if coverage_method == "api_coverage_only" and line_percent is None else "source_line_coverage_available"
    if selected_function.get("available"):
        low_count = len(low_coverage_apis)
        low_applicability = "source_line_coverage_available"
    report = {
        "schema_version": "2.0",
        "target_library": target_library,
        "target_version": target_version,
        "main_coverage_number": round(main_number, 4),
        "main_coverage_label": main_label,
        "coverage_method": coverage_method,
        "coverage_scope": summary.get("coverage_scope", ""),
        "coverage_metric": "API execution coverage over accepted API subset" if coverage_method == "api_coverage_only" else main_label,
        "api_execution_coverage_percent": api_coverage_percent,
        "api_coverage_percent": api_coverage_percent,
        "covered_api_count": covered_api_count,
        "total_api_count": total_api_count,
        "function_line_coverage_percent": round(selected_percent, 4) if selected_percent is not None else None,
        "selected_function_coverage": selected_function,
        "python_line_coverage_percent": round(python_line_percent, 4) if python_line_percent is not None else None,
        "native_line_coverage_percent": round(native_line_percent, 4) if native_line_percent is not None else None,
        "line_coverage_percent": round(line_percent, 4) if line_percent is not None else None,
        "total_covered_lines": covered_lines if coverable_lines else None,
        "total_coverable_lines": coverable_lines if coverable_lines else None,
        "apis_with_low_coverage": low_count,
        "low_coverage_applicability": low_applicability,
        "low_coverage_apis": low_coverage_apis,
        "coverage_available": {
            "api_execution": True,
            "selected_function": bool(selected_function.get("available")),
            "python": python_line_percent is not None,
            "native": native_line_percent is not None,
        },
        "coverage_summary": {
            "api_execution": {
                "available": True,
                "covered": covered_api_count,
                "total": total_api_count,
                "percent": api_coverage_percent,
            },
            "selected_function_line_coverage": {
                "available": bool(selected_function.get("available")),
                "covered": int(selected_function.get("covered_lines", 0) or 0),
                "total": int(selected_function.get("coverable_lines", 0) or 0),
                "percent": round(selected_percent, 4) if selected_percent is not None else None,
                "unavailable_apis": selected_function.get("excluded_unavailable_apis", []),
            },
            "python_line_coverage": {
                "available": python_line_percent is not None,
                "covered": int(_numeric(python_campaign.get("python_covered_lines")) or 0) if python_campaign else 0,
                "total": int(_numeric(python_campaign.get("python_coverable_lines")) or 0) if python_campaign else 0,
                "percent": round(python_line_percent, 4) if python_line_percent is not None else None,
            },
            "native_line_coverage": {
                "available": native_line_percent is not None,
                "covered": int(_numeric(native_campaign.get("native_covered_lines")) or 0) if native_campaign else 0,
                "total": int(_numeric(native_campaign.get("native_coverable_lines")) or 0) if native_campaign else 0,
                "percent": round(native_line_percent, 4) if native_line_percent is not None else None,
            },
        },
        "unique_valid_programs": int(summary.get("unique_valid_programs", 0) or 0),
        "total_valid_programs": int(summary.get("total_valid_programs", summary.get("valid_programs", 0)) or 0),
        "execution_time_seconds": round(execution_time_seconds, 4),
        "seed": summary.get("seed", ""),
        "mutation_budget": summary.get("mutation_budget", ""),
        "run_id": run_id or str(summary.get("run_id", "") or ""),
        "environment_metadata": environment_metadata,
        "per_api": per_api,
        "coverage_limitations": list(dict.fromkeys(limitations)),
        "limitations": list(dict.fromkeys(limitations)),
        "coverage_warnings": list(dict.fromkeys(summary.get("coverage_warnings", []) or [])),
        "oracle_summary": {
            "device_oracle_enabled": bool(summary.get("device_oracle_enabled")),
            "device_oracle_devices": list(summary.get("device_oracle_devices", []) or []),
            "device_oracle_mismatches": int(summary.get("device_oracle_mismatch", 0) or 0),
            "edge_oracle_mutations": bool(summary.get("edge_oracle_mutations")),
        },
    }
    return report


def write_coverage_markdown(path: str, report: Dict[str, Any], bug_report: Dict[str, Any]) -> None:
    lines = [
        "# DeepFuzz Stage 4 Coverage Summary",
        "",
        f"- Target: {report.get('target_library', 'unknown')} {report.get('target_version', '')}".rstrip(),
        f"- Coverage metric: {report.get('main_coverage_number')} ({report.get('main_coverage_label')})",
        f"- Method: {report.get('coverage_method')}",
        f"- API execution coverage: {report.get('covered_api_count')}/{report.get('total_api_count')} ({report.get('api_coverage_percent')}%)",
        f"- Unique valid programs: {report.get('unique_valid_programs')}",
        f"- Candidate bugs: {bug_report.get('summary', {}).get('total_candidates', 0)}",
        f"- APIs below low-coverage threshold: {report.get('apis_with_low_coverage')}",
        f"- Execution time: {report.get('execution_time_seconds')} seconds",
    ]
    if report.get("coverage_method") == "api_coverage_only":
        lines.append(f"- Interpretation: {report.get('api_coverage_percent')}% API execution coverage over the accepted API subset; source-line coverage unavailable in this configuration.")
        lines.append("- Python line coverage: unavailable")
    if report.get("line_coverage_percent") is not None:
        lines.append(f"- Line coverage: {report.get('line_coverage_percent')}% ({report.get('total_covered_lines')}/{report.get('total_coverable_lines')} lines)")
    if report.get("function_line_coverage_percent") is not None:
        lines.append(f"- Selected-function line coverage: {report.get('function_line_coverage_percent')}%")
    elif report.get("coverage_available", {}).get("python"):
        lines.append("- Selected-function line coverage: unavailable for resolved Python source spans")
    limitations = report.get("coverage_limitations") or []
    if limitations:
        lines.extend(["", "## Limitations"])
        lines.extend([f"- {item}" for item in limitations])
    atomic_write_text(path, "\n".join(lines) + "\n")


def run_stage4(
    init_dir: str,
    results_dir: str,
    mutation_budget: int,
    seed: int,
    ok_csv: str = "",
    coverage_scope: str = "both",
    enable_python_coverage: bool = False,
    python_cov_source: Sequence[str] | None = None,
    python_cov_omit: Sequence[str] | None = None,
    native_coverage_engine: str = "none",
    native_source_root: str = "",
    native_build_dir: str = "",
    gcovr_executable: str = "gcovr",
    gcov_executable: str = "",
    gcovr_filter: Sequence[str] | None = None,
    gcovr_exclude: Sequence[str] | None = None,
    gcovr_extra_args: Sequence[str] | None = None,
    native_html: bool = False,
    gcovr_jobs: int = 1,
    worker_timeout_sec: int = 120,
    case_timeout_sec: int = 30,
    low_coverage_threshold: float = 60.0,
    only_api_list: str = "",
    limit: int = 0,
    known_bugs_file: str = "",
    unresolved_failures_file: str = "",
    run_id: str = "",
    merge_unselected: bool = False,
    enable_device_oracle: bool = False,
    device_oracle_devices: Sequence[str] | None = None,
    device_oracle_rtol: float = 1e-4,
    device_oracle_atol: float = 1e-5,
    edge_oracle_mutations: bool = False,
) -> None:
    started = time.perf_counter()
    edge_oracle_mutations = bool(edge_oracle_mutations or enable_device_oracle)
    ensure_dir(results_dir)
    requested_coverage_scope = coverage_scope
    if coverage_scope == "api_only":
        coverage_scope = "none"
    all_init_paths = list(iter_init_objects(init_dir, ok_csv=ok_csv, only_api_list=only_api_list))
    init_paths = list(all_init_paths[:limit]) if limit and limit > 0 else all_init_paths
    selected_api_names = {api_name_from_init(path) for path in init_paths}

    coverage_warnings: List[str] = []
    native_requested_by_scope = coverage_scope in {"native", "per_api", "campaign", "both"}
    if coverage_scope == "native" and native_coverage_engine == "none":
        raise RuntimeError("--coverage-scope native requires --native-coverage-engine auto/gcovr")
    if native_coverage_engine == "auto":
        has_gcov = bool(native_build_dir and os.path.isdir(native_build_dir) and any(Path(native_build_dir).rglob("*.gcno")))
        native_coverage_engine = "gcovr" if has_gcov and shutil.which(gcovr_executable) else "none"
    if coverage_scope == "native" and native_coverage_engine == "none":
        raise RuntimeError("--coverage-scope native requires GCOV/LCOV instrumentation; use --coverage-scope both/python for fallback metrics")
    if native_requested_by_scope and native_coverage_engine == "none":
        coverage_warnings.append("coverage_scope requested native coverage, but no native coverage engine/source build is configured; native coverage is marked unavailable.")

    per_api_python = coverage_scope == "python" or (coverage_scope in {"per_api", "both"} and enable_python_coverage)
    per_api_native = coverage_scope == "native" or (coverage_scope in {"per_api", "both"} and native_coverage_engine != "none")
    campaign_python = coverage_scope in {"campaign", "both"} and enable_python_coverage
    campaign_native = coverage_scope in {"campaign", "both"} and native_coverage_engine != "none"
    # Keep each API execution in its own subprocess.  A single multi-API
    # campaign worker can be killed by one native TensorFlow abort and lose the
    # rest of the run, so union line coverage is exported only when native tools
    # can aggregate data across isolated per-API subprocesses.
    run_per_api = coverage_scope in {"none", "python", "native", "per_api", "both", "campaign"}
    run_campaign = False

    native_cov = NativeCoverageSession(
        enabled=per_api_native or campaign_native,
        engine=native_coverage_engine,
        source_root=native_source_root,
        build_dir=native_build_dir,
        gcovr_executable=gcovr_executable,
        gcov_executable=gcov_executable,
        gcovr_filter=gcovr_filter,
        gcovr_exclude=gcovr_exclude,
        extra_args=gcovr_extra_args,
        emit_html=native_html,
        gcovr_jobs=gcovr_jobs,
    )
    native_cov.validate_or_raise()

    summary: Dict[str, Any] = {
        "selected_before_limit": len(all_init_paths),
        "selected_apis": len(init_paths),
        "limit": limit,
        "executed_apis": 0,
        "materialization_fail": 0,
        "base_success": 0,
        "base_fail": 0,
        "mutation_cases": 0,
        "mutation_success": 0,
        "mutation_fail": 0,
        "negative_expected_fail": 0,
        "expected_negative_rejection": 0,
        "negative_accepted_not_bug": 0,
        "invalid_valid_mutation": 0,
        "device_oracle_mismatch": 0,
        "valid_programs": 0,
        "total_valid_programs": 0,
        "unique_valid_programs": 0,
        "apis_with_failures": 0,
        "apis_with_nan": 0,
        "apis_with_low_coverage": 0,
        "coverage_scope": requested_coverage_scope if requested_coverage_scope == "api_only" else coverage_scope,
        "python_coverage_enabled": bool(per_api_python or campaign_python),
        "native_coverage_engine": native_coverage_engine,
        "worker_timeout_sec": worker_timeout_sec,
        "case_timeout_sec": case_timeout_sec,
        "low_coverage_threshold": low_coverage_threshold,
        "seed": seed,
        "mutation_budget": mutation_budget,
        "run_id": run_id,
        "low_coverage_applicability": "source_line_coverage_available" if (per_api_native or campaign_native or campaign_python) else "not_applicable_source_line_coverage_unavailable",
        "coverage_warnings": coverage_warnings,
        "device_oracle_enabled": bool(enable_device_oracle),
        "device_oracle_devices": list(device_oracle_devices or []),
        "device_oracle_rtol": float(device_oracle_rtol),
        "device_oracle_atol": float(device_oracle_atol),
        "edge_oracle_mutations": bool(edge_oracle_mutations),
    }

    results_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    low_cov_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    execution_dir = ensure_dir(os.path.join(results_dir, "execution_results"))
    worker_dir = ensure_dir(os.path.join(results_dir, "worker_results"))
    per_api_cov_dir = ensure_dir(os.path.join(results_dir, "coverage_reports", "per_api"))
    campaign_cov_dir = ensure_dir(os.path.join(results_dir, "coverage_reports", "campaign"))

    if run_per_api:
        for idx, init_path in enumerate(init_paths):
            api_name = api_name_from_init(init_path)
            print(f"[stage4] {idx + 1}/{len(init_paths)} start {api_name}", flush=True)
            api_safe = safe_name(api_name)
            worker_json = os.path.join(worker_dir, f"{api_safe}.worker.json")
            py_dir = os.path.join(per_api_cov_dir, "python", api_safe)
            if per_api_native:
                native_cov.reset()
            worker = launch_worker(
                init_paths=[init_path],
                output_json=worker_json,
                mutation_budget=mutation_budget,
                seed=seed + idx,
                timeout_sec=worker_timeout_sec,
                enable_python_coverage=per_api_python,
                python_cov_source=python_cov_source or [],
                python_cov_omit=python_cov_omit or [],
                python_cov_dir=py_dir,
                case_timeout_sec=case_timeout_sec,
                enable_device_oracle=enable_device_oracle,
                device_oracle_devices=device_oracle_devices or [],
                device_oracle_rtol=device_oracle_rtol,
                device_oracle_atol=device_oracle_atol,
                edge_oracle_mutations=edge_oracle_mutations,
            )
            bundles = worker.get("bundles") if isinstance(worker.get("bundles"), list) else []
            if bundles:
                bundle = bundles[0]
            else:
                err = worker.get("stderr") or worker.get("load_error") or "worker produced no result"
                bundle = worker_failure_bundle(init_path, seed + idx, int(worker.get("exit_code", -1)), bool(worker.get("timeout")), str(err))
            bundle["worker_stdout"] = worker.get("stdout", "")
            bundle["worker_stderr"] = worker.get("stderr", "")
            results_json_path = os.path.join(execution_dir, f"{api_safe}.results.json")
            native_summary = native_cov.export(os.path.join(per_api_cov_dir, "native", api_safe, "coverage")) if per_api_native else {"enabled": False}
            append_bundle_outputs(
                bundle=bundle,
                results_json_path=results_json_path,
                results_rows=results_rows,
                failure_rows=failure_rows,
                summary=summary,
                worker_exit_code=int(worker.get("exit_code", -1)),
                worker_timeout=bool(worker.get("timeout")),
            )
            coverage_row = make_coverage_row(
                api_full_name=api_name,
                scope="per_api" if coverage_scope not in {"none", "python", "native"} else coverage_scope,
                py_summary=worker.get("python_coverage", {"enabled": False}),
                native_summary=native_summary,
                init_json_path=init_path,
                results_json_path=results_json_path,
                worker=worker,
            )
            coverage_rows.append(coverage_row)
            if per_api_native or campaign_native or campaign_python:
                low_hit, low_reason = evaluate_low_coverage(coverage_row, low_coverage_threshold)
                if low_hit:
                    summary["apis_with_low_coverage"] += 1
                    low_cov_rows.append({"api_full_name": api_name, "reason": low_reason, **coverage_row})
            latest = results_rows[-1] if results_rows else {}
            print(
                f"[stage4] {idx + 1}/{len(init_paths)} done {api_name} "
                f"exit={worker.get('exit_code')} timeout={worker.get('timeout')} "
                f"base_success={latest.get('base_success', '')}",
                flush=True,
            )

    if per_api_python:
        py_summary = export_combined_python_coverage(
            per_api_python_dir=os.path.join(per_api_cov_dir, "python"),
            campaign_python_dir=os.path.join(campaign_cov_dir, "python"),
            python_cov_source=python_cov_source or [],
            python_cov_omit=python_cov_omit or [],
        )
        coverage_row = make_coverage_row(
            api_full_name="__CAMPAIGN__",
            scope="campaign",
            py_summary=py_summary,
            native_summary={"enabled": False},
            worker={"exit_code": "", "timeout": ""},
        )
        coverage_rows.append(coverage_row)
        write_json(os.path.join(campaign_cov_dir, "python_coverage_summary.json"), coverage_row)

    if run_campaign:
        campaign_worker_json = os.path.join(worker_dir, "__campaign__.worker.json")
        if campaign_native:
            native_cov.reset()
        worker = launch_worker(
            init_paths=init_paths,
            output_json=campaign_worker_json,
            mutation_budget=mutation_budget,
            seed=seed + 100000,
            timeout_sec=worker_timeout_sec * max(1, len(init_paths)),
            enable_python_coverage=campaign_python,
            python_cov_source=python_cov_source or [],
            python_cov_omit=python_cov_omit or [],
            python_cov_dir=os.path.join(campaign_cov_dir, "python"),
            case_timeout_sec=case_timeout_sec,
            enable_device_oracle=enable_device_oracle,
            device_oracle_devices=device_oracle_devices or [],
            device_oracle_rtol=device_oracle_rtol,
            device_oracle_atol=device_oracle_atol,
            edge_oracle_mutations=edge_oracle_mutations,
        )
        native_summary = native_cov.export(os.path.join(campaign_cov_dir, "native", "coverage")) if campaign_native else {"enabled": False}
        coverage_row = make_coverage_row(
            api_full_name="__CAMPAIGN__",
            scope="campaign",
            py_summary=worker.get("python_coverage", {"enabled": False}),
            native_summary=native_summary,
            worker=worker,
        )
        coverage_rows.append(coverage_row)
        write_json(os.path.join(campaign_cov_dir, "campaign_summary.json"), coverage_row)
        if not run_per_api:
            bundles = worker.get("bundles") if isinstance(worker.get("bundles"), list) else []
            by_api = {str(b.get("api_full_name", "")): b for b in bundles if isinstance(b, dict)}
            for idx, init_path in enumerate(init_paths):
                api_name = api_name_from_init(init_path)
                bundle = by_api.get(api_name) or worker_failure_bundle(
                    init_path,
                    seed + 100000 + idx,
                    int(worker.get("exit_code", -1)),
                    bool(worker.get("timeout")),
                    str(worker.get("stderr") or worker.get("load_error") or "campaign worker produced no result"),
                )
                bundle["worker_stdout"] = worker.get("stdout", "")
                bundle["worker_stderr"] = worker.get("stderr", "")
                results_json_path = os.path.join(execution_dir, f"{safe_name(api_name)}.results.json")
                append_bundle_outputs(
                    bundle=bundle,
                    results_json_path=results_json_path,
                    results_rows=results_rows,
                    failure_rows=failure_rows,
                    summary=summary,
                    worker_exit_code=int(worker.get("exit_code", -1)),
                    worker_timeout=bool(worker.get("timeout")),
                )

    if only_api_list and merge_unselected:
        results_rows = merge_rows_preserving_unselected(
            read_csv_rows(os.path.join(results_dir, "results.csv")),
            results_rows,
            selected_api_names,
            key_fields=("api_full_name",),
        )
        failure_rows = merge_rows_preserving_unselected(
            read_csv_rows(os.path.join(results_dir, "failures.csv")),
            failure_rows,
            selected_api_names,
            key_fields=("api_full_name", "stage", "mutation_intent", "param", "rule", "error_type"),
        )
        coverage_rows = merge_rows_preserving_unselected(
            read_csv_rows(os.path.join(results_dir, "coverage_report.csv")),
            coverage_rows,
            selected_api_names,
            key_fields=("api_full_name", "coverage_scope"),
        )
        low_cov_rows = merge_rows_preserving_unselected(
            read_csv_rows(os.path.join(results_dir, "low_coverage.csv")),
            low_cov_rows,
            selected_api_names,
            key_fields=("api_full_name", "reason"),
        )
        recompute_summary_from_merged_rows(summary, results_rows, failure_rows)

    finalize_valid_program_counts(summary)
    write_json(os.path.join(results_dir, "summary.json"), summary)
    write_csv(os.path.join(results_dir, "results.csv"), results_rows, [
        "api_full_name", "import_path", "backend", "library_version", "init_json_path", "seed",
        "base_success", "base_error", "mutation_cases", "mutation_success", "mutation_fail",
        "negative_expected_fail", "expected_negative_rejection", "negative_accepted_not_bug", "invalid_valid_mutation",
        "device_oracle_mismatch", "valid_programs", "nan_observed", "worker_exit_code", "worker_timeout", "results_json_path",
    ])
    write_csv(os.path.join(results_dir, "failures.csv"), failure_rows, [
        "api_full_name", "stage", "mutation_intent", "param", "rule", "error_type", "error",
        "worker_exit_code", "worker_timeout", "results_json_path",
    ])
    write_csv(os.path.join(results_dir, "low_coverage.csv"), low_cov_rows, [
        "api_full_name", "reason", "coverage_scope", "python_statement_percent", "python_branch_percent",
        "python_covered_lines", "python_coverable_lines", "native_line_percent", "native_covered_lines",
        "native_coverable_lines", "native_branch_percent", "native_function_percent", "python_json_path",
        "native_json_path", "python_export_error", "native_export_error", "worker_exit_code",
        "worker_timeout", "init_json_path", "results_json_path",
    ])
    write_csv(os.path.join(results_dir, "coverage_report.csv"), coverage_rows, [
        "api_full_name", "coverage_scope", "python_statement_percent", "python_branch_percent",
        "python_covered_lines", "python_coverable_lines", "native_line_percent", "native_covered_lines",
        "native_coverable_lines", "native_branch_percent", "native_function_percent", "python_json_path",
        "native_json_path", "python_export_error", "native_export_error", "worker_exit_code",
        "worker_timeout", "init_json_path", "results_json_path",
    ])
    execution_rows, audit_rows = build_execution_and_audit_rows(results_rows)
    write_csv(os.path.join(results_dir, "execution_summary.csv"), execution_rows, EXECUTION_SUMMARY_FIELDS)
    write_csv(os.path.join(results_dir, "failure.csv"), audit_rows, FAILURE_AUDIT_FIELDS)

    elapsed = time.perf_counter() - started
    summary["execution_time_seconds"] = round(elapsed, 4)
    bug_report = build_bug_report(
        results_dir,
        results_rows,
        mutation_budget=mutation_budget,
        known_bugs_file=known_bugs_file,
        unresolved_failures_file=unresolved_failures_file,
    )
    write_json(os.path.join(results_dir, "bug_report.json"), bug_report)
    write_bug_report_csv(os.path.join(results_dir, "bug_report.csv"), bug_report)
    write_bug_audit_csv(os.path.join(results_dir, "bug_audit.csv"), bug_report)
    coverage_report = build_coverage_report(
        summary=summary,
        coverage_rows=coverage_rows,
        results_rows=results_rows,
        bug_report=bug_report,
        execution_time_seconds=elapsed,
        native_requested=native_coverage_engine != "none",
        python_requested=bool(enable_python_coverage),
        run_id=run_id,
    )
    if coverage_report.get("coverage_method") == "api_coverage_only" and coverage_report.get("line_coverage_percent") is None:
        summary["apis_with_low_coverage"] = None
        summary["low_coverage_applicability"] = "not_applicable_source_line_coverage_unavailable"
        coverage_report["apis_with_low_coverage"] = None
        coverage_report["low_coverage_applicability"] = summary["low_coverage_applicability"]
    else:
        summary["apis_with_low_coverage"] = coverage_report.get("apis_with_low_coverage", summary.get("apis_with_low_coverage", 0))
        summary["low_coverage_applicability"] = coverage_report.get("low_coverage_applicability", summary.get("low_coverage_applicability", ""))
        low_cov_rows = [
            {
                "api_full_name": row.get("api_full_name", ""),
                "reason": row.get("reason", ""),
                "coverage_scope": "selected_function",
                "python_statement_percent": "",
                "python_branch_percent": "",
                "python_covered_lines": row.get("function_covered_lines", ""),
                "python_coverable_lines": row.get("function_coverable_lines", ""),
                "native_line_percent": "",
                "native_covered_lines": "",
                "native_coverable_lines": "",
                "native_branch_percent": "",
                "native_function_percent": "",
                "python_json_path": coverage_report.get("selected_function_coverage", {}).get("python_json_path", ""),
                "native_json_path": "",
                "python_export_error": "",
                "native_export_error": "",
                "worker_exit_code": "",
                "worker_timeout": "",
                "init_json_path": "",
                "results_json_path": "",
            }
            for row in coverage_report.get("low_coverage_apis", [])
            if isinstance(row, dict)
        ]
        write_csv(os.path.join(results_dir, "low_coverage.csv"), low_cov_rows, [
            "api_full_name", "reason", "coverage_scope", "python_statement_percent", "python_branch_percent",
            "python_covered_lines", "python_coverable_lines", "native_line_percent", "native_covered_lines",
            "native_coverable_lines", "native_branch_percent", "native_function_percent", "python_json_path",
            "native_json_path", "python_export_error", "native_export_error", "worker_exit_code",
            "worker_timeout", "init_json_path", "results_json_path",
        ])
    bug_report["environment_metadata"] = coverage_report.get("environment_metadata", {})
    bug_report["coverage_method"] = coverage_report.get("coverage_method", "")
    refresh_bug_summary(bug_report)
    write_json(os.path.join(results_dir, "bug_report.json"), bug_report)
    write_bug_report_csv(os.path.join(results_dir, "bug_report.csv"), bug_report)
    write_bug_audit_csv(os.path.join(results_dir, "bug_audit.csv"), bug_report)
    write_json(os.path.join(results_dir, "coverage_report.json"), coverage_report)
    write_json(os.path.join(results_dir, "selected_function_coverage.json"), coverage_report.get("selected_function_coverage", {}))
    write_coverage_markdown(os.path.join(results_dir, "coverage_report.md"), coverage_report, bug_report)
    write_json(os.path.join(results_dir, "summary.json"), summary)

    print(
        f"[stage4] selected={summary['selected_apis']} executed={summary['executed_apis']} "
        f"base_success={summary['base_success']} base_fail={summary['base_fail']} "
        f"mutation_success={summary['mutation_success']} mutation_fail={summary['mutation_fail']} "
        f"low_cov={summary['apis_with_low_coverage']}"
    )
    print(f"[stage4] results: {os.path.join(results_dir, 'results.csv')}")
    print(f"[stage4] failures: {os.path.join(results_dir, 'failures.csv')}")
    print(f"[stage4] coverage_report: {os.path.join(results_dir, 'coverage_report.json')}")
    print(f"[stage4] bug_report: {os.path.join(results_dir, 'bug_report.json')}")
    print(f"[stage4] summary: {os.path.join(results_dir, 'summary.json')}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 4 runner: isolated mutation execution and coverage export")
    ap.add_argument("--init-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ok-csv", default="")
    ap.add_argument("--mutation-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--only-api-list", default="", help="Optional newline-delimited api_full_name allowlist")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on selected APIs after filtering; 0 means no limit")
    ap.add_argument("--coverage-scope", choices=["none", "api_only", "python", "native", "per_api", "campaign", "both"], default="both")
    ap.add_argument("--low-coverage-threshold", type=float, default=60.0)
    ap.add_argument("--worker-timeout-sec", type=int, default=120)
    ap.add_argument("--case-timeout-sec", type=int, default=30)
    ap.add_argument("--enable-python-coverage", action="store_true")
    ap.add_argument("--python-cov-source", action="append", default=[])
    ap.add_argument("--python-cov-omit", action="append", default=[])
    ap.add_argument("--native-coverage-engine", choices=["none", "auto", "gcovr"], default="auto")
    ap.add_argument("--native-source-root", default="")
    ap.add_argument("--native-build-dir", default="")
    ap.add_argument("--native-html", action="store_true")
    ap.add_argument("--gcovr-executable", default="gcovr")
    ap.add_argument("--gcov-executable", default="")
    ap.add_argument("--gcovr-filter", action="append", default=[])
    ap.add_argument("--gcovr-exclude", action="append", default=[])
    ap.add_argument("--gcovr-extra-arg", action="append", default=[])
    ap.add_argument("--gcovr-jobs", type=int, default=1)
    ap.add_argument("--known-bugs-file", default="", help="Optional JSON file with known bug signatures to mark as known_existing.")
    ap.add_argument("--unresolved-failures-csv", default="", help="Canonical repair-loop CSV of final unresolved APIs to include in bug_report.")
    ap.add_argument("--merge-unselected", action="store_true", help="Merge prior rows for APIs outside --only-api-list; intended for targeted repair reruns.")
    ap.add_argument("--enable-device-oracle", action="store_true", default=os.environ.get("ENABLE_DEVICE_ORACLE", "").strip().lower() in {"1", "true", "yes"}, help="Compare valid executions across CPU and accelerator devices when available.")
    ap.add_argument("--device-oracle-device", action="append", default=[], help="Device selector for the differential oracle; use cpu, accelerator, cuda, gpu, mps, xpu, or backend device labels. Defaults to cpu+accelerator.")
    ap.add_argument("--device-oracle-rtol", type=float, default=float(os.environ.get("DEVICE_ORACLE_RTOL", "1e-4")))
    ap.add_argument("--device-oracle-atol", type=float, default=float(os.environ.get("DEVICE_ORACLE_ATOL", "1e-5")))
    ap.add_argument("--edge-oracle-mutations", action="store_true", default=os.environ.get("EDGE_ORACLE_MUTATIONS", "").strip().lower() in {"1", "true", "yes"}, help="Add NaN/Inf/signed-zero/large-value valid mutations so the device oracle can find silent numerical divergences.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_stage4(
            init_dir=args.init_dir,
            results_dir=args.results_dir,
            mutation_budget=args.mutation_budget,
            seed=args.seed,
            ok_csv=args.ok_csv,
            coverage_scope=args.coverage_scope,
            enable_python_coverage=args.enable_python_coverage,
            python_cov_source=args.python_cov_source,
            python_cov_omit=args.python_cov_omit,
            native_coverage_engine=args.native_coverage_engine,
            native_source_root=args.native_source_root,
            native_build_dir=args.native_build_dir,
            gcovr_executable=args.gcovr_executable,
            gcov_executable=args.gcov_executable,
            gcovr_filter=args.gcovr_filter,
            gcovr_exclude=args.gcovr_exclude,
            gcovr_extra_args=args.gcovr_extra_arg,
            native_html=args.native_html,
            gcovr_jobs=args.gcovr_jobs,
            worker_timeout_sec=args.worker_timeout_sec,
            case_timeout_sec=args.case_timeout_sec,
            low_coverage_threshold=args.low_coverage_threshold,
            only_api_list=args.only_api_list,
            limit=args.limit,
            known_bugs_file=args.known_bugs_file,
            unresolved_failures_file=args.unresolved_failures_csv,
            run_id=args.run_id,
            merge_unselected=args.merge_unselected,
            enable_device_oracle=args.enable_device_oracle,
            device_oracle_devices=args.device_oracle_device,
            device_oracle_rtol=args.device_oracle_rtol,
            device_oracle_atol=args.device_oracle_atol,
            edge_oracle_mutations=args.edge_oracle_mutations,
        )
    except RuntimeError as exc:
        print(f"[stage4] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
