#!/usr/bin/env python3
from __future__ import annotations

"""Optional known-bug calibration seeds for the Stage 4 oracle.

These are deliberately separate from normal fuzz-discovered bug reports.  They
exercise oracle categories and backend skip handling.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.result_io import atomic_write_json


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_id: str
    name: str
    api_surface: List[str]
    expected_category: str
    backend: str
    required_env: Dict[str, str] = field(default_factory=dict)
    advanced: bool = False
    optional_reason: str = ""


BENCHMARKS: List[BenchmarkDefinition] = [
    BenchmarkDefinition(
        "B1",
        "shard_map + grad wrong result",
        ["jax.shard_map", "jax.grad", "jax.vmap"],
        "wrong_result_on_valid_input",
        "cpu_multi_device",
        {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
    ),
    BenchmarkDefinition(
        "B2",
        "batched jnp.linalg.solve performance regression",
        ["jax.numpy.linalg.solve", "jax.grad", "jax.jit"],
        "performance_regression",
        "cpu",
        optional_reason="requires a versioned baseline and machine-noise budget",
    ),
    BenchmarkDefinition(
        "B3",
        "bool sum closure constant wrong result",
        ["jax.numpy.sum", "jax.jit"],
        "wrong_result_on_valid_input",
        "gpu",
    ),
    BenchmarkDefinition(
        "B4",
        "scan + vmap wrong result",
        ["jax.lax.scan", "jax.vmap"],
        "wrong_result_on_valid_input",
        "gpu",
    ),
    BenchmarkDefinition(
        "B5",
        "debug.breakpoint + cond + vmap tracer error",
        ["jax.debug.breakpoint", "jax.lax.cond", "jax.vmap"],
        "unexpected_exception_on_valid_input",
        "any",
        advanced=True,
        optional_reason="side-effectful debug semantics are version-sensitive",
    ),
    BenchmarkDefinition(
        "B6",
        "custom_jvp + custom_transpose tracer error",
        ["jax.custom_jvp", "custom_transpose", "jax.jvp", "jax.value_and_grad"],
        "unexpected_exception_on_valid_input",
        "any",
        advanced=True,
        optional_reason="advanced autodiff calibration seed",
    ),
]


def _import_jax() -> Tuple[Any, str]:
    try:
        import jax  # type: ignore

        return jax, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _device_platforms(jax: Any) -> List[str]:
    try:
        return [str(getattr(device, "platform", "") or "").lower() for device in jax.devices()]
    except Exception:
        return []


def backend_skip_reason(defn: BenchmarkDefinition, jax: Any) -> str:
    if jax is None:
        return "jax import unavailable"
    platforms = _device_platforms(jax)
    if defn.backend == "gpu" and "gpu" not in platforms:
        return "GPU backend unavailable"
    if defn.backend == "cpu" and "cpu" not in platforms:
        return "CPU backend unavailable"
    if defn.backend == "cpu_multi_device" and platforms.count("cpu") < 2:
        return "CPU simulated multi-device backend unavailable; set JAX_PLATFORMS=cpu and XLA_FLAGS=--xla_force_host_platform_device_count=2 before importing JAX"
    return ""


def _run_b1(jax: Any) -> Dict[str, Any]:
    # The concrete shard_map repro is version-sensitive.  We run only when the
    # required surface exists; otherwise this remains a structured skip rather
    # than a fabricated pass.
    missing = [name for name in ["shard_map", "grad", "vmap"] if not hasattr(jax, name)]
    if missing:
        return {"status": "skipped", "skip_reason": f"missing JAX APIs: {', '.join(missing)}"}
    return {"status": "skipped", "skip_reason": "B1 calibration body requires repository-pinned JAX repro values"}


def _run_optional_skip(defn: BenchmarkDefinition) -> Dict[str, Any]:
    return {"status": "skipped", "skip_reason": defn.optional_reason or "optional calibration seed not enabled for this backend"}


RUNNERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "B1": _run_b1,
}


def run_benchmark(defn: BenchmarkDefinition) -> Dict[str, Any]:
    started = time.perf_counter()
    jax, import_reason = _import_jax()
    skip = import_reason or backend_skip_reason(defn, jax)
    if skip:
        status = {"status": "skipped", "skip_reason": skip}
    elif defn.benchmark_id in RUNNERS:
        status = RUNNERS[defn.benchmark_id](jax)
    else:
        status = _run_optional_skip(defn)
    return {
        "schema_version": "1.0",
        **asdict(defn),
        **status,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 4),
    }


def run_known_bug_benchmarks(outdir: str, selected: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    selected_set = {s for s in (selected or []) if s}
    defs = [d for d in BENCHMARKS if not selected_set or d.benchmark_id in selected_set]
    results = [run_benchmark(defn) for defn in defs]
    payload = {
        "schema_version": "1.0",
        "kind": "known_bug_calibration_seeds",
        "generated_at_unix": int(time.time()),
        "results": results,
        "summary": {
            "total": len(results),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "matched_expected_signal": sum(1 for r in results if r.get("status") == "matched_expected_signal"),
        },
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "known_bug_benchmarks.json", payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run optional JAX known-bug calibration seeds")
    ap.add_argument("--outdir", default="stage4/results/jax-known-bug-benchmarks")
    ap.add_argument("--benchmark", action="append", default=[], help="Benchmark id to run, e.g. B1. Omit for all.")
    args = ap.parse_args(argv)
    payload = run_known_bug_benchmarks(args.outdir, selected=args.benchmark)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
