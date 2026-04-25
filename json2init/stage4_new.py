#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

# Allow running this file from outside the repo while still importing the existing helpers.
ROOT = Path.cwd()
for candidate in [ROOT, ROOT / "json2init"]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

from deepfuzz_common import clone_value, import_api, load_json, mutate_value, resolve_python_object, summarize_python_value  # type: ignore


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



def iter_init_objects(init_dir: str, ok_csv: str = "") -> Iterable[str]:
    allow = load_ok_init_paths(ok_csv) if ok_csv else None
    for path in sorted(Path(init_dir).glob("*.init.json")):
        full = os.path.abspath(str(path))
        if allow is not None and allow and full not in allow:
            continue
        yield str(path)



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
    }
    api_path = str(init_obj.get("import_path", "") or "")
    t0 = time.perf_counter()
    try:
        api = import_api(api_path)
        result = api(**kwargs)
        out["success"] = True
        out["result_summary"] = summarize_python_value(result)
    except Exception as exc:
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)
    finally:
        out["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
    return out



def choose_mutations(meta: Dict[str, Any], budget: int, rng: random.Random) -> List[str]:
    rules = [str(x) for x in (meta.get("mutation_rules") or []) if str(x)]
    if not rules:
        return []
    rng.shuffle(rules)
    if budget <= len(rules):
        return rules[:budget]
    out: List[str] = []
    while len(out) < budget:
        shuffled = list(rules)
        rng.shuffle(shuffled)
        out.extend(shuffled)
    return out[:budget]



def summarize_mutation_failure(api_name: str, param: str, rule: str, exc: Exception) -> Dict[str, Any]:
    return {
        "api_full_name": api_name,
        "param": param,
        "rule": rule,
        "success": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }



def fuzz_one(init_path: str, mutation_budget: int, rng: random.Random, failures_only: bool) -> Dict[str, Any]:
    init_obj = load_json(init_path)
    api_name = str(init_obj.get("api_full_name", Path(init_path).stem))
    result_bundle: Dict[str, Any] = {
        "api_full_name": api_name,
        "init_json_path": init_path,
        "base_execution": {},
        "mutations": [],
        "materialization_errors": [],
    }
    if not init_obj.get("ready_for_stage4"):
        result_bundle["materialization_errors"] = list(init_obj.get("readiness_reasons", []))
        return result_bundle

    kwargs, reasons = materialize_kwargs(init_obj)
    result_bundle["materialization_errors"] = reasons
    if reasons:
        return result_bundle

    base_result = execute_api(init_obj, kwargs)
    result_bundle["base_execution"] = base_result
    if not base_result.get("success"):
        return result_bundle

    params = init_obj.get("params", {}) if isinstance(init_obj.get("params"), dict) else {}
    for name, meta in params.items():
        if name not in kwargs:
            continue
        for rule in choose_mutations(meta, mutation_budget, rng):
            mutated_kwargs = {k: clone_value(v) for k, v in kwargs.items()}
            try:
                mutated_kwargs[name] = mutate_value(mutated_kwargs[name], rule, rng)
            except Exception as exc:
                failure = summarize_mutation_failure(api_name, name, rule, exc)
                if (not failures_only) or (not failure["success"]):
                    result_bundle["mutations"].append(failure)
                continue
            exec_result = execute_api(init_obj, mutated_kwargs)
            exec_result.update({"param": name, "rule": rule})
            if failures_only and exec_result.get("success"):
                continue
            result_bundle["mutations"].append(exec_result)
    return result_bundle



def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def run_fuzz(init_dir: str, results_dir: str, mutation_budget: int, seed: int, ok_csv: str = "", failures_only: bool = False) -> None:
    os.makedirs(results_dir, exist_ok=True)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    summary = {
        "total": 0,
        "base_success": 0,
        "base_fail": 0,
        "mutation_success": 0,
        "mutation_fail": 0,
        "apis_with_failures": 0,
    }
    failures_jsonl = os.path.join(results_dir, "failures.jsonl")
    if os.path.exists(failures_jsonl):
        os.remove(failures_jsonl)

    for init_path in iter_init_objects(init_dir, ok_csv=ok_csv):
        summary["total"] += 1
        bundle = fuzz_one(init_path, mutation_budget=mutation_budget, rng=rng, failures_only=failures_only)
        api_name = str(bundle.get("api_full_name", ""))
        base_ok = bool(bundle.get("base_execution", {}).get("success"))
        summary["base_success" if base_ok else "base_fail"] += 1

        bundle_has_failure = False
        if bundle.get("materialization_errors"):
            bundle_has_failure = True
        if bundle.get("base_execution") and not base_ok:
            bundle_has_failure = True

        mut_count = 0
        for m in bundle.get("mutations", []):
            mut_count += 1
            if m.get("success"):
                summary["mutation_success"] += 1
            else:
                summary["mutation_fail"] += 1
                bundle_has_failure = True
                with open(failures_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"api_full_name": api_name, **m}, ensure_ascii=False) + "\n")

        if bundle.get("materialization_errors"):
            with open(failures_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "api_full_name": api_name,
                    "stage": "materialize",
                    "errors": bundle.get("materialization_errors", []),
                }, ensure_ascii=False) + "\n")

        if bundle.get("base_execution") and not base_ok:
            with open(failures_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "api_full_name": api_name,
                    "stage": "base_execution",
                    **bundle.get("base_execution", {}),
                }, ensure_ascii=False) + "\n")

        if bundle_has_failure:
            summary["apis_with_failures"] += 1
            out_path = os.path.join(results_dir, f"{api_name.replace('.', '_')}.results.json")
            write_json(out_path, bundle)
            results_json_path = out_path
        else:
            results_json_path = ""

        rows.append({
            "api_full_name": api_name,
            "base_success": str(base_ok),
            "base_error": bundle.get("base_execution", {}).get("error", ""),
            "recorded_mutations": str(mut_count),
            "results_json_path": results_json_path,
        })

    write_json(os.path.join(results_dir, "summary.json"), summary)
    with open(os.path.join(results_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["api_full_name", "base_success", "base_error", "recorded_mutations", "results_json_path"])
        w.writeheader()
        w.writerows(rows)
    print(
        f"[stage4] total={summary['total']} base_success={summary['base_success']} base_fail={summary['base_fail']} "
        f"mutation_success={summary['mutation_success']} mutation_fail={summary['mutation_fail']}"
    )



def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4: mutate and execute runtime init objects")
    ap.add_argument("--init-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--coverage-dir", default="")  # kept for compatibility; not used internally anymore
    ap.add_argument("--mutation-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ok-csv", default="")
    ap.add_argument("--failures-only", action="store_true")
    args = ap.parse_args()
    run_fuzz(
        init_dir=args.init_dir,
        results_dir=args.results_dir,
        mutation_budget=args.mutation_budget,
        seed=args.seed,
        ok_csv=args.ok_csv,
        failures_only=args.failures_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
