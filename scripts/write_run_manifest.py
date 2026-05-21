#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return ""


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "") or "")
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a reproducibility manifest for a thesis run")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--api-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default=os.environ.get("SEED", "1337"))
    ap.add_argument("--mutation-budget", default=os.environ.get("MUTATION_BUDGET", "8"))
    ap.add_argument("--coverage-scope", default=os.environ.get("COVERAGE_SCOPE", "python"))
    ap.add_argument("--native-coverage", default=os.environ.get("NATIVE_COVERAGE_STATUS", "unavailable_on_this_run"))
    args = ap.parse_args()

    api_list = Path(args.api_list)
    if not api_list.is_absolute():
        api_list = ROOT / api_list
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    payload: Dict[str, Any] = {
        "library": args.lib,
        "run_id": args.run_id,
        "api_list": str(api_list),
        "api_list_sha256": _sha256(api_list) if api_list.exists() else "",
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": {
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "tensorflow": _version("tensorflow"),
            "coverage": _version("coverage"),
        },
        "ollama_models": {
            "general": os.environ.get("DEEPFUZZ_MODEL", ""),
            "fast": os.environ.get("DEEPFUZZ_FAST_MODEL", ""),
            "repair": os.environ.get("DEEPFUZZ_REPAIR_MODEL", ""),
            "strong_repair": os.environ.get("DEEPFUZZ_STRONG_REPAIR_MODEL", ""),
        },
        "seed": int(float(args.seed or 0)),
        "mutation_budget": int(float(args.mutation_budget or 0)),
        "coverage_scope": args.coverage_scope,
        "python_cov_source": os.environ.get("PYTHON_COV_SOURCE", args.lib),
        "native_coverage": args.native_coverage,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
