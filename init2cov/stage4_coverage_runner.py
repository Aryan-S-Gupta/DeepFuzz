#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path.cwd()
for candidate in [ROOT, ROOT / "json2init", ROOT / "stage4"]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

from deepfuzz_common import (  # type: ignore
    clone_value,
    import_api,
    load_json,
    mutate_value,
    resolve_python_object,
    summarize_python_value,
)

try:
    from coverage import Coverage
except Exception:
    Coverage = None

try:
    import pandas as pd
except Exception:
    pd = None

_ILLEGAL_XLSX_CHARS_RE = __import__("re").compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


# --------------------------- filesystem helpers ---------------------------

def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in str(name))


def write_json(path: str | Path, data: Any) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _clean_xlsx_value(v: Any) -> Any:
    if isinstance(v, str):
        return _ILLEGAL_XLSX_CHARS_RE.sub("", v)
    return v


def write_xlsx_if_possible(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    if pd is None:
        return
    df = pd.DataFrame(list(rows))
    df = df.apply(lambda col: col.map(_clean_xlsx_value))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    df.to_excel(path, index=False)


# --------------------------- selection helpers ---------------------------

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


def iter_init_objects(init_dir: str, ok_csv: str = "", only_api_list: str = "") -> Iterable[str]:
    allow_paths = load_ok_init_paths(ok_csv) if ok_csv else None
    allow_apis = load_api_filter(only_api_list) if only_api_list else None
    env_retry = os.environ.get("STAGE4_RETRY_LIST", "").strip()
    if env_retry and not only_api_list:
        allow_apis = load_api_filter(env_retry)
    for path in sorted(Path(init_dir).glob("*.init.json")):
        full = os.path.abspath(str(path))
        if allow_paths is not None and allow_paths and full not in allow_paths:
            continue
        api_full_name = path.stem.replace(".init", "")
        if allow_apis is not None and allow_apis and api_full_name not in allow_apis:
            continue
        yield str(path)


# --------------------------- execution helpers ---------------------------

def materialize_kwargs(init_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    kwargs: Dict[str, Any] = {}
    reasons: List[str] = []
    backend = str(init_obj.get("backend", "python") or "python")
    for name, meta in (init_obj.get("params") or {}).items():
        if not isinstance(meta, dict):
            reasons.append(f"{name}: invalid runtime param object")
            continue
        include = bool(meta.get("include_in_base_call"))
        spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else {}
        default_text = str(spec.get("default", "") or "").strip()
        if not include and default_text:
            continue
        if not include and str(spec.get("flag", "") or "") == "Optional":
            continue
        unresolved = str(meta.get("unresolved_reason", "") or "").strip()
        if unresolved:
            reasons.append(f"{name}: {unresolved}")
            continue
        try:
            kwargs[name] = resolve_python_object(meta.get("base_seed_spec", {}), backend=backend)
        except Exception as exc:
            reasons.append(f"{name}: materialization failed: {type(exc).__name__}: {exc}")
    return kwargs, reasons


def execute_api(init_obj: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error_type": "",
        "error": "",
        "result_summary": {},
        "duration_ms": 0.0,
        "has_nan": False,
    }
    api_path = str(init_obj.get("import_path", "") or "")
    t0 = time.perf_counter()
    try:
        api = import_api(api_path)
        result = api(**kwargs)
        result_summary = summarize_python_value(result)
        out["success"] = True
        out["result_summary"] = result_summary
        out["has_nan"] = bool(result_summary.get("has_nan", False))
    except Exception as exc:
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)
    finally:
        out["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
    return out


SIZE_RULES = {"shape_expand", "shape_shrink", "structure_grow", "structure_shrink"}
TYPE_RULES = {"dtype_mutate", "type_widen", "type_switch"}
VALUE_RULES = {"value_noise", "value_mask", "value_division", "scalar_delta", "element_delta", "string_replace", "string_empty", "enum_switch"}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def choose_mutations(meta: Dict[str, Any], budget: int, rng: random.Random) -> List[str]:
    rules = [str(x) for x in (meta.get("mutation_rules") or []) if str(x)]
    if not rules or budget <= 0:
        return []
    chosen_type = str(meta.get("chosen_type", "") or "").lower()
    # Thesis mode: guarantee broad mutation-family coverage whenever possible.
    triad: List[str] = []
    for bucket in (SIZE_RULES, TYPE_RULES, VALUE_RULES):
        candidates = [r for r in rules if r in bucket]
        if candidates:
            triad.append(rng.choice(candidates))
    # For tensor/structured/numeric inputs, prefer family coverage first.
    prefer_triad = bool(triad) and chosen_type in {"tensor", "list", "tuple", "sequence", "int", "float", "number", "string", "bool", "any", "dict", "dtype", "device"}
    ordered_pool = list(rules)
    rng.shuffle(ordered_pool)
    if prefer_triad:
        ordered = _dedupe_keep_order(triad + ordered_pool)
    else:
        ordered = ordered_pool
    if budget <= len(ordered):
        return ordered[:budget]
    out: List[str] = []
    while len(out) < budget:
        block = list(ordered)
        rng.shuffle(block)
        out.extend(block)
    return out[:budget]


def summarize_mutation_failure(api_name: str, param: str, rule: str, exc: Exception) -> Dict[str, Any]:
    return {
        "api_full_name": api_name,
        "param": param,
        "rule": rule,
        "success": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "duration_ms": 0.0,
        "result_summary": {},
        "has_nan": False,
    }


def build_mutated_cases(init_obj: Dict[str, Any], kwargs: Dict[str, Any], mutation_budget: int, rng: random.Random) -> List[Dict[str, Any]]:
    mutations: List[Dict[str, Any]] = []
    params = init_obj.get("params", {}) if isinstance(init_obj.get("params"), dict) else {}
    api_name = str(init_obj.get("api_full_name", ""))
    for name, meta in params.items():
        if name not in kwargs:
            continue
        for rule in choose_mutations(meta, mutation_budget, rng):
            mutated_kwargs = {k: clone_value(v) for k, v in kwargs.items()}
            try:
                mutated_kwargs[name] = mutate_value(mutated_kwargs[name], rule, rng)
            except Exception as exc:
                mutations.append(summarize_mutation_failure(api_name, name, rule, exc))
                continue
            exec_result = execute_api(init_obj, mutated_kwargs)
            exec_result.update({"param": name, "rule": rule})
            mutations.append(exec_result)
    return mutations


def fuzz_one(init_path: str, mutation_budget: int, rng: random.Random) -> Dict[str, Any]:
    init_obj = load_json(init_path)
    api_name = str(init_obj.get("api_full_name", Path(init_path).stem.replace(".init", "")))
    result_bundle: Dict[str, Any] = {
        "api_full_name": api_name,
        "init_json_path": init_path,
        "base_execution": {},
        "mutations": [],
        "materialization_errors": [],
        "base_kwargs_materialized": False,
    }
    if not init_obj.get("ready_for_stage4"):
        result_bundle["materialization_errors"] = list(init_obj.get("readiness_reasons", []))
        return result_bundle

    kwargs, reasons = materialize_kwargs(init_obj)
    result_bundle["materialization_errors"] = reasons
    result_bundle["base_kwargs_materialized"] = not bool(reasons)
    if reasons:
        return result_bundle

    base_result = execute_api(init_obj, kwargs)
    result_bundle["base_execution"] = base_result
    if not base_result.get("success"):
        return result_bundle

    result_bundle["mutations"] = build_mutated_cases(init_obj, kwargs, mutation_budget, rng)
    return result_bundle


# --------------------------- python coverage ---------------------------

class PythonCoverageSession:
    def __init__(self, enabled: bool, source: Sequence[str], omit: Sequence[str], base_dir: str) -> None:
        self.enabled = enabled and Coverage is not None
        self.source = list(source)
        self.omit = list(omit)
        self.base_dir = base_dir
        self._cov: Any = None
        ensure_dir(base_dir)

    def start(self, data_file: str) -> None:
        if not self.enabled:
            return
        self._cov = Coverage(data_file=data_file, branch=True, source=self.source or None, omit=self.omit or None)
        self._cov.erase()
        self._cov.start()

    def stop_and_export(self, data_file: str, json_path: str, html_dir: str, keep_html: bool = True) -> Dict[str, Any]:
        if not self.enabled or self._cov is None:
            return {"enabled": False}
        summary: Dict[str, Any] = {"enabled": True, "data_file": data_file}
        try:
            self._cov.stop()
            self._cov.save()
            ensure_dir(os.path.dirname(json_path))
            self._cov.json_report(outfile=json_path, pretty_print=True)
            summary["json_path"] = json_path
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            totals = data.get("totals", {}) if isinstance(data, dict) else {}
            if isinstance(totals, dict):
                summary["statement_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["covered_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["num_statements"] = totals.get("num_statements", "")
                summary["covered_lines"] = totals.get("covered_lines", totals.get("covered_statements", ""))
                summary["missing_lines"] = totals.get("missing_lines", "")
                summary["num_branches"] = totals.get("num_branches", "")
                summary["covered_branches"] = totals.get("covered_branches", "")
                summary["missing_branches"] = totals.get("missing_branches", "")
                nb = totals.get("num_branches", 0) or 0
                cb = totals.get("covered_branches", 0) or 0
                if nb:
                    summary["branch_percent"] = round(100.0 * float(cb) / float(nb), 4)
            if keep_html:
                self._cov.html_report(directory=html_dir)
                summary["html_dir"] = html_dir
        except Exception as exc:
            summary["export_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._cov = None
        return summary

    def combine_and_export(self, data_files: Sequence[str], json_path: str, html_dir: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        merged_data_file = os.path.join(self.base_dir, ".coverage.campaign")
        cov = Coverage(data_file=merged_data_file, branch=True, source=self.source or None, omit=self.omit or None)
        cov.erase()
        try:
            cov.combine(list(data_files), strict=False)
            cov.save()
        except Exception as exc:
            return {"enabled": True, "combine_error": f"{type(exc).__name__}: {exc}"}
        self._cov = cov
        return self.stop_and_export(merged_data_file, json_path, html_dir, keep_html=True)


# --------------------------- native coverage ---------------------------

class NativeCoverageSession:
    def __init__(
        self,
        enabled: bool,
        engine: str,
        source_root: str,
        build_dir: str,
        reports_dir: str,
        gcovr_executable: str = "gcovr",
        gcov_executable: str = "",
        gcovr_filter: Sequence[str] | None = None,
        gcovr_exclude: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
        emit_html: bool = False,
        gcovr_jobs: int = 1,
    ) -> None:
        self.enabled = enabled and engine == "gcovr" and bool(source_root) and bool(build_dir)
        self.engine = engine
        self.source_root = os.path.abspath(source_root) if source_root else ""
        self.build_dir = os.path.abspath(build_dir) if build_dir else ""
        self.reports_dir = reports_dir
        self.gcovr_executable = gcovr_executable
        self.gcov_executable = gcov_executable
        self.gcovr_filter = list(gcovr_filter or [])
        self.gcovr_exclude = list(gcovr_exclude or [])
        self.extra_args = list(extra_args or [])
        self.emit_html = bool(emit_html)
        self.gcovr_jobs = max(int(gcovr_jobs or 1), 1)
        ensure_dir(reports_dir)

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
        base_cmd = [
            self.gcovr_executable,
            "-r", self.source_root,
            "--object-directory", self.build_dir,
            "-j", str(self.gcovr_jobs),
            "--json-summary-pretty",
            "--json-summary", out_json,
        ]
        if self.gcov_executable:
            base_cmd += ["--gcov-executable", self.gcov_executable]
        for filt in self.gcovr_filter:
            base_cmd += ["--filter", filt]
        for exc in self.gcovr_exclude:
            base_cmd += ["--exclude", exc]
        base_cmd += list(self.extra_args)
        summary: Dict[str, Any] = {
            "enabled": True,
            "engine": "gcovr",
            "summary_json": out_json,
            "html_path": out_html,
        }
        try:
            subprocess.run(base_cmd, cwd=self.build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            summary["export_error"] = exc.stderr or exc.stdout or str(exc)
            return summary
        if self.emit_html:
            html_cmd = [
                self.gcovr_executable,
                "-r", self.source_root,
                "--object-directory", self.build_dir,
                "-j", str(self.gcovr_jobs),
                "--html", "--html-details",
                "-o", out_html,
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
            summary["line_percent"] = data.get("line_percent", "")
            summary["line_covered"] = data.get("line_covered", "")
            summary["line_total"] = data.get("line_total", "")
            summary["branch_percent"] = data.get("branch_percent", "")
            summary["branch_covered"] = data.get("branch_covered", "")
            summary["branch_total"] = data.get("branch_total", "")
            summary["function_percent"] = data.get("function_percent", "")
            summary["function_covered"] = data.get("function_covered", "")
            summary["function_total"] = data.get("function_total", "")
        except Exception as exc:
            summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        return summary


# --------------------------- metrics helpers ---------------------------

def numeric_or_none(v: Any) -> Optional[float]:
    try:
        if v in {None, "", "nan", "NaN"}:
            return None
        return float(v)
    except Exception:
        return None


def evaluate_low_coverage(summary: Dict[str, Any], threshold: float) -> Tuple[bool, str]:
    metrics = []
    for key in ("line_percent", "branch_percent", "function_percent", "covered_percent", "statement_percent"):
        val = numeric_or_none(summary.get(key))
        if val is not None:
            metrics.append((key, val))
    if not metrics:
        return False, ""
    low = [(k, v) for k, v in metrics if v < threshold]
    if not low:
        return False, ""
    return True, "; ".join(f"{k}={v:.2f}" for k, v in low)


def compute_bundle_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
    muts = bundle.get("mutations", []) if isinstance(bundle.get("mutations"), list) else []
    base_success = bool(base.get("success"))
    mutation_count = len(muts)
    mutation_success = sum(1 for m in muts if isinstance(m, dict) and m.get("success"))
    mutation_fail = mutation_count - mutation_success
    nan_observed = bool(base.get("has_nan", False)) or any(bool(m.get("has_nan", False)) for m in muts if isinstance(m, dict))
    return {
        "base_success": base_success,
        "mutation_count": mutation_count,
        "mutation_success": mutation_success,
        "mutation_fail": mutation_fail,
        "nan_observed": nan_observed,
    }


# --------------------------- passes ---------------------------

def run_single_api_with_optional_coverage(
    init_path: str,
    mutation_budget: int,
    rng: random.Random,
    python_cov: Optional[PythonCoverageSession],
    native_cov: Optional[NativeCoverageSession],
    per_api_reports_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    api_name = Path(init_path).stem.replace(".init", "")
    api_safe = safe_name(api_name)
    py_summary: Dict[str, Any] = {}
    native_summary: Dict[str, Any] = {}

    if native_cov and native_cov.enabled:
        native_cov.reset()

    py_data_file = os.path.join(per_api_reports_dir, "python", api_safe, ".coverage")
    py_json = os.path.join(per_api_reports_dir, "python", api_safe, "coverage.json")
    py_html = os.path.join(per_api_reports_dir, "python", api_safe, "html")
    if python_cov and python_cov.enabled:
        ensure_dir(os.path.dirname(py_data_file))
        python_cov.start(py_data_file)

    bundle = fuzz_one(init_path, mutation_budget=mutation_budget, rng=rng)

    if python_cov and python_cov.enabled:
        py_summary = python_cov.stop_and_export(py_data_file, py_json, py_html)

    if native_cov and native_cov.enabled:
        out_prefix = os.path.join(per_api_reports_dir, "native", api_safe, "coverage")
        native_summary = native_cov.export(out_prefix)

    return bundle, py_summary, native_summary


def campaign_export(
    python_cov: Optional[PythonCoverageSession],
    python_data_files: Sequence[str],
    native_cov: Optional[NativeCoverageSession],
    campaign_reports_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    py_summary: Dict[str, Any] = {}
    native_summary: Dict[str, Any] = {}
    if python_cov and python_cov.enabled:
        py_summary = python_cov.combine_and_export(
            python_data_files,
            os.path.join(campaign_reports_dir, "python", "coverage.json"),
            os.path.join(campaign_reports_dir, "python", "html"),
        )
    if native_cov and native_cov.enabled:
        native_summary = native_cov.export(os.path.join(campaign_reports_dir, "native", "coverage"))
    return py_summary, native_summary


# --------------------------- stage 4 runner ---------------------------

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
    low_coverage_threshold: float = 60.0,
    only_api_list: str = "",
    limit: int = 0,
) -> None:
    ensure_dir(results_dir)
    all_init_paths = list(iter_init_objects(init_dir, ok_csv=ok_csv, only_api_list=only_api_list))
    init_paths = list(all_init_paths[:limit]) if limit and limit > 0 else all_init_paths
    master_rng = random.Random(seed)

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
        "apis_with_failures": 0,
        "apis_with_nan": 0,
        "apis_with_low_coverage": 0,
        "coverage_scope": coverage_scope,
        "python_coverage_enabled": enable_python_coverage,
        "native_coverage_engine": native_coverage_engine,
        "low_coverage_threshold": low_coverage_threshold,
    }

    results_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    low_cov_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    python_cov_data_files_campaign: List[str] = []

    per_api_enabled = coverage_scope in {"per_api", "both"}
    campaign_enabled = coverage_scope in {"campaign", "both"}

    per_api_reports_dir = ensure_dir(os.path.join(results_dir, "coverage_reports", "per_api"))
    campaign_reports_dir = ensure_dir(os.path.join(results_dir, "coverage_reports", "campaign"))
    execution_dir = ensure_dir(os.path.join(results_dir, "execution_results"))

    # ---------------- per-api pass ----------------
    if per_api_enabled:
        py_cov = PythonCoverageSession(enable_python_coverage, python_cov_source or [], python_cov_omit or [], os.path.join(results_dir, ".python_cov_per_api"))
        native_cov = NativeCoverageSession(
            enabled=(native_coverage_engine == "gcovr"),
            engine=native_coverage_engine,
            source_root=native_source_root,
            build_dir=native_build_dir,
            reports_dir=per_api_reports_dir,
            gcovr_executable=gcovr_executable,
            gcov_executable=gcov_executable,
            gcovr_filter=gcovr_filter,
            gcovr_exclude=gcovr_exclude,
            extra_args=gcovr_extra_args,
            emit_html=native_html,
            gcovr_jobs=gcovr_jobs,
        )

        for idx, init_path in enumerate(init_paths):
            bundle_rng = random.Random(master_rng.randint(0, 10**9) + idx)
            bundle, py_summary, native_summary = run_single_api_with_optional_coverage(
                init_path=init_path,
                mutation_budget=mutation_budget,
                rng=bundle_rng,
                python_cov=py_cov,
                native_cov=native_cov,
                per_api_reports_dir=per_api_reports_dir,
            )
            api_name = str(bundle.get("api_full_name", ""))
            init_json_path = str(bundle.get("init_json_path", init_path))
            metrics = compute_bundle_metrics(bundle)
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
            if metrics["nan_observed"]:
                summary["apis_with_nan"] += 1

            results_json_path = os.path.join(execution_dir, f"{safe_name(api_name)}.results.json")
            write_json(results_json_path, bundle)

            bundle_has_failure = False
            if bundle.get("materialization_errors"):
                bundle_has_failure = True
                failure_rows.append({
                    "api_full_name": api_name,
                    "stage": "materialize",
                    "error_type": "materialization_error",
                    "error": " | ".join(bundle.get("materialization_errors", [])),
                    "results_json_path": results_json_path,
                })
            base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
            if base and not metrics["base_success"]:
                bundle_has_failure = True
                failure_rows.append({
                    "api_full_name": api_name,
                    "stage": "base_execution",
                    "error_type": base.get("error_type", ""),
                    "error": base.get("error", ""),
                    "results_json_path": results_json_path,
                })

            for m in bundle.get("mutations", []):
                if not isinstance(m, dict):
                    continue
                if not m.get("success"):
                    bundle_has_failure = True
                    failure_rows.append({
                        "api_full_name": api_name,
                        "stage": f"mutation:{m.get('param', '')}:{m.get('rule', '')}",
                        "error_type": m.get("error_type", ""),
                        "error": m.get("error", ""),
                        "results_json_path": results_json_path,
                    })

            if bundle_has_failure:
                summary["apis_with_failures"] += 1

            if py_summary.get("enabled"):
                python_cov_data_files_campaign.append(str(py_summary.get("data_file", "")))

            results_row = {
                "api_full_name": api_name,
                "init_json_path": init_json_path,
                "base_success": str(metrics["base_success"]),
                "base_error": base.get("error", ""),
                "mutation_cases": metrics["mutation_count"],
                "mutation_success": metrics["mutation_success"],
                "mutation_fail": metrics["mutation_fail"],
                "nan_observed": metrics["nan_observed"],
                "results_json_path": results_json_path,
            }
            results_rows.append(results_row)

            coverage_row = {
                "api_full_name": api_name,
                "coverage_scope": "per_api",
                "python_statement_percent": py_summary.get("statement_percent", py_summary.get("covered_percent", "")),
                "python_branch_percent": py_summary.get("branch_percent", ""),
                "native_line_percent": native_summary.get("line_percent", ""),
                "native_branch_percent": native_summary.get("branch_percent", ""),
                "native_function_percent": native_summary.get("function_percent", ""),
                "python_json_path": py_summary.get("json_path", ""),
                "native_json_path": native_summary.get("summary_json", ""),
                "python_export_error": py_summary.get("export_error", py_summary.get("combine_error", "")),
                "native_export_error": native_summary.get("export_error", native_summary.get("parse_error", native_summary.get("html_error", ""))),
                "init_json_path": init_json_path,
                "results_json_path": results_json_path,
            }
            coverage_rows.append(coverage_row)

            low_hit, low_reason = evaluate_low_coverage({
                "line_percent": native_summary.get("line_percent"),
                "branch_percent": native_summary.get("branch_percent") or py_summary.get("branch_percent"),
                "function_percent": native_summary.get("function_percent"),
                "statement_percent": py_summary.get("statement_percent", py_summary.get("covered_percent")),
                "covered_percent": py_summary.get("covered_percent"),
            }, threshold=low_coverage_threshold)
            if low_hit:
                summary["apis_with_low_coverage"] += 1
                low_cov_rows.append({
                    "api_full_name": api_name,
                    "reason": low_reason,
                    "python_statement_percent": py_summary.get("statement_percent", py_summary.get("covered_percent", "")),
                    "python_branch_percent": py_summary.get("branch_percent", ""),
                    "native_line_percent": native_summary.get("line_percent", ""),
                    "native_branch_percent": native_summary.get("branch_percent", ""),
                    "native_function_percent": native_summary.get("function_percent", ""),
                })

    # ---------------- campaign pass ----------------
    if campaign_enabled:
        native_cov_campaign = NativeCoverageSession(
            enabled=(native_coverage_engine == "gcovr"),
            engine=native_coverage_engine,
            source_root=native_source_root,
            build_dir=native_build_dir,
            reports_dir=campaign_reports_dir,
            gcovr_executable=gcovr_executable,
            gcov_executable=gcov_executable,
            gcovr_filter=gcovr_filter,
            gcovr_exclude=gcovr_exclude,
            extra_args=gcovr_extra_args,
            emit_html=native_html,
            gcovr_jobs=gcovr_jobs,
        )
        py_cov_campaign = PythonCoverageSession(enable_python_coverage, python_cov_source or [], python_cov_omit or [], os.path.join(results_dir, ".python_cov_campaign"))

        if native_cov_campaign.enabled:
            native_cov_campaign.reset()

        campaign_data_files: List[str] = []
        for idx, init_path in enumerate(init_paths):
            api_name = Path(init_path).stem.replace(".init", "")
            api_safe = safe_name(api_name)
            data_file = os.path.join(results_dir, ".python_cov_campaign", api_safe, ".coverage")
            if py_cov_campaign.enabled:
                ensure_dir(os.path.dirname(data_file))
                py_cov_campaign.start(data_file)
            bundle = fuzz_one(init_path, mutation_budget=mutation_budget, rng=random.Random(master_rng.randint(0, 10**9) + 100000 + idx))
            if py_cov_campaign.enabled:
                py_cov_campaign.stop_and_export(
                    data_file,
                    os.path.join(results_dir, ".python_cov_campaign", api_safe, "coverage.json"),
                    os.path.join(results_dir, ".python_cov_campaign", api_safe, "html"),
                    keep_html=False,
                )
                campaign_data_files.append(data_file)

            if not per_api_enabled:
                api_name = str(bundle.get("api_full_name", api_name))
                init_json_path = str(bundle.get("init_json_path", init_path))
                metrics = compute_bundle_metrics(bundle)
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
                if metrics["nan_observed"]:
                    summary["apis_with_nan"] += 1

                results_json_path = os.path.join(execution_dir, f"{safe_name(api_name)}.results.json")
                write_json(results_json_path, bundle)

                bundle_has_failure = False
                if bundle.get("materialization_errors"):
                    bundle_has_failure = True
                    failure_rows.append({
                        "api_full_name": api_name,
                        "stage": "materialize",
                        "error_type": "materialization_error",
                        "error": " | ".join(bundle.get("materialization_errors", [])),
                        "results_json_path": results_json_path,
                    })
                base = bundle.get("base_execution", {}) if isinstance(bundle.get("base_execution"), dict) else {}
                if base and not metrics["base_success"]:
                    bundle_has_failure = True
                    failure_rows.append({
                        "api_full_name": api_name,
                        "stage": "base_execution",
                        "error_type": base.get("error_type", ""),
                        "error": base.get("error", ""),
                        "results_json_path": results_json_path,
                    })
                for m in bundle.get("mutations", []):
                    if not isinstance(m, dict):
                        continue
                    if not m.get("success"):
                        bundle_has_failure = True
                        failure_rows.append({
                            "api_full_name": api_name,
                            "stage": f"mutation:{m.get('param', '')}:{m.get('rule', '')}",
                            "error_type": m.get("error_type", ""),
                            "error": m.get("error", ""),
                            "results_json_path": results_json_path,
                        })
                if bundle_has_failure:
                    summary["apis_with_failures"] += 1

                results_rows.append({
                    "api_full_name": api_name,
                    "init_json_path": init_json_path,
                    "base_success": str(metrics["base_success"]),
                    "base_error": base.get("error", ""),
                    "mutation_cases": metrics["mutation_count"],
                    "mutation_success": metrics["mutation_success"],
                    "mutation_fail": metrics["mutation_fail"],
                    "nan_observed": metrics["nan_observed"],
                    "results_json_path": results_json_path,
                })

        py_campaign_summary, native_campaign_summary = campaign_export(
            python_cov=py_cov_campaign,
            python_data_files=campaign_data_files,
            native_cov=native_cov_campaign,
            campaign_reports_dir=campaign_reports_dir,
        )
        coverage_rows.append({
            "api_full_name": "__CAMPAIGN__",
            "coverage_scope": "campaign",
            "python_statement_percent": py_campaign_summary.get("statement_percent", py_campaign_summary.get("covered_percent", "")),
            "python_branch_percent": py_campaign_summary.get("branch_percent", ""),
            "native_line_percent": native_campaign_summary.get("line_percent", ""),
            "native_branch_percent": native_campaign_summary.get("branch_percent", ""),
            "native_function_percent": native_campaign_summary.get("function_percent", ""),
            "python_json_path": py_campaign_summary.get("json_path", ""),
            "native_json_path": native_campaign_summary.get("summary_json", ""),
            "python_export_error": py_campaign_summary.get("export_error", py_campaign_summary.get("combine_error", "")),
            "native_export_error": native_campaign_summary.get("export_error", native_campaign_summary.get("parse_error", native_campaign_summary.get("html_error", ""))),
            "init_json_path": "",
            "results_json_path": "",
        })
        write_json(os.path.join(campaign_reports_dir, "campaign_summary.json"), coverage_rows[-1])

    # ---------------- outputs ----------------
    write_json(os.path.join(results_dir, "summary.json"), summary)
    write_csv(os.path.join(results_dir, "results.csv"), results_rows, [
        "api_full_name", "init_json_path", "base_success", "base_error", "mutation_cases", "mutation_success", "mutation_fail", "nan_observed", "results_json_path"
    ])
    write_csv(os.path.join(results_dir, "failures.csv"), failure_rows, [
        "api_full_name", "stage", "error_type", "error", "results_json_path"
    ])
    write_csv(os.path.join(results_dir, "low_coverage.csv"), low_cov_rows, [
        "api_full_name", "reason", "python_statement_percent", "python_branch_percent", "native_line_percent", "native_branch_percent", "native_function_percent"
    ])
    write_csv(os.path.join(results_dir, "coverage_report.csv"), coverage_rows, [
        "api_full_name", "coverage_scope", "python_statement_percent", "python_branch_percent",
        "native_line_percent", "native_branch_percent", "native_function_percent",
        "python_json_path", "native_json_path", "python_export_error", "native_export_error", "init_json_path", "results_json_path"
    ])
    write_xlsx_if_possible(os.path.join(results_dir, "coverage_report.xlsx"), coverage_rows)

    retry_api_names = sorted({
        str(row["api_full_name"])
        for row in (failure_rows + low_cov_rows)
        if str(row.get("api_full_name", "")) and str(row.get("api_full_name")) != "__CAMPAIGN__"
    })
    retry_list_path = os.path.join(results_dir, "retry_api_list.txt")
    with open(retry_list_path, "w", encoding="utf-8") as f:
        for name in retry_api_names:
            f.write(name + "\n")

    retry_script = os.path.join(results_dir, "coverage_retry.sh")
    cmd = [
        "python3", os.path.relpath(__file__, start=os.getcwd()),
        "--init-dir", init_dir,
        "--results-dir", results_dir,
        "--mutation-budget", str(mutation_budget),
        "--seed", str(seed),
        "--coverage-scope", coverage_scope,
        "--low-coverage-threshold", str(low_coverage_threshold),
        "--only-api-list", retry_list_path,
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    if ok_csv:
        cmd += ["--ok-csv", ok_csv]
    if enable_python_coverage:
        cmd += ["--enable-python-coverage"]
        for src in (python_cov_source or []):
            cmd += ["--python-cov-source", src]
        for omit in (python_cov_omit or []):
            cmd += ["--python-cov-omit", omit]
    if native_coverage_engine != "none":
        cmd += ["--native-coverage-engine", native_coverage_engine, "--native-source-root", native_source_root, "--native-build-dir", native_build_dir]
        if gcovr_executable:
            cmd += ["--gcovr-executable", gcovr_executable]
        if gcov_executable:
            cmd += ["--gcov-executable", gcov_executable]
        for filt in (gcovr_filter or []):
            cmd += ["--gcovr-filter", filt]
        for exc in (gcovr_exclude or []):
            cmd += ["--gcovr-exclude", exc]
        for extra in (gcovr_extra_args or []):
            cmd += ["--gcovr-extra-arg", extra]
    with open(retry_script, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        if retry_api_names:
            f.write(" ".join(subprocess.list2cmdline([arg]) if " " in arg else arg for arg in cmd) + "\n")
        else:
            f.write("echo 'No failed or low-coverage APIs remaining.'\n")
    os.chmod(retry_script, 0o755)

    print(
        f"[stage4-thesis] selected={summary['selected_apis']} executed={summary['executed_apis']} "
        f"base_success={summary['base_success']} base_fail={summary['base_fail']} "
        f"mutation_success={summary['mutation_success']} mutation_fail={summary['mutation_fail']} low_cov={summary['apis_with_low_coverage']}"
    )
    print(f"[stage4-thesis] results: {os.path.join(results_dir, 'results.csv')}")
    print(f"[stage4-thesis] failures: {os.path.join(results_dir, 'failures.csv')}")
    print(f"[stage4-thesis] low_coverage: {os.path.join(results_dir, 'low_coverage.csv')}")
    print(f"[stage4-thesis] coverage_report: {os.path.join(results_dir, 'coverage_report.csv')}")
    print(f"[stage4-thesis] retry_script: {retry_script}")
    print(f"[stage4-thesis] summary: {os.path.join(results_dir, 'summary.json')}")


# --------------------------- CLI ---------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 4 thesis runner: materialize, mutate, execute, and export coverage")
    ap.add_argument("--init-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ok-csv", default="")
    ap.add_argument("--mutation-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only-api-list", default="", help="Optional newline-delimited api_full_name allowlist")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of selected APIs after filtering; 0 means no limit")

    ap.add_argument("--coverage-scope", choices=["per_api", "campaign", "both"], default="both")
    ap.add_argument("--low-coverage-threshold", type=float, default=60.0)

    ap.add_argument("--enable-python-coverage", action="store_true")
    ap.add_argument("--python-cov-source", action="append", default=[])
    ap.add_argument("--python-cov-omit", action="append", default=[])

    ap.add_argument("--native-coverage-engine", choices=["none", "gcovr"], default="none")
    ap.add_argument("--native-source-root", default="")
    ap.add_argument("--native-build-dir", default="")
    ap.add_argument("--gcovr-executable", default="gcovr")
    ap.add_argument("--gcov-executable", default="")
    ap.add_argument("--gcovr-filter", action="append", default=[])
    ap.add_argument("--gcovr-exclude", action="append", default=[])
    ap.add_argument("--gcovr-extra-arg", action="append", default=[])
    return ap


def main() -> int:
    args = build_parser().parse_args()
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
        low_coverage_threshold=args.low_coverage_threshold,
        only_api_list=args.only_api_list,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
