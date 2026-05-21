#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.result_io import atomic_write_text  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_api_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Select APIs below a selected-function coverage threshold for targeted reruns")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "latest"))
    ap.add_argument("--stage4-results-dir", default="")
    ap.add_argument("--api-list", default="")
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("LOW_COVERAGE_THRESHOLD", "10")))
    ap.add_argument("--include-unavailable", action="store_true", help="Also include APIs whose Python source span was unavailable")
    ap.add_argument("--max-apis", type=int, default=0, help="Optional cap on selected low-coverage APIs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = ROOT / "pipeline_runs" / args.lib / args.run_id
    stage4_dir = Path(args.stage4_results_dir) if args.stage4_results_dir else ROOT / "stage4" / "results" / f"{args.lib}-coverage"
    if not stage4_dir.is_absolute():
        stage4_dir = ROOT / stage4_dir
    api_list_path = Path(args.api_list) if args.api_list else run_dir / "api_list.txt"
    if not api_list_path.is_absolute():
        api_list_path = ROOT / api_list_path
    out_path = Path(args.out) if args.out else run_dir / "repair" / "low_coverage_api_list.txt"
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    allowed = _load_api_list(api_list_path)
    report = _load_json(stage4_dir / "coverage_report.json")
    per_api = report.get("per_api", []) if isinstance(report.get("per_api"), list) else []
    selected: List[str] = []
    reasons: List[str] = []
    for item in per_api:
        if not isinstance(item, dict):
            continue
        api = str(item.get("api", "") or "").strip()
        if not api:
            continue
        if allowed and api not in allowed:
            continue
        percent = _float_or_none(item.get("function_line_coverage_percent"))
        unavailable = not bool(item.get("function_coverage_available", percent is not None))
        if percent is not None and percent < args.threshold:
            selected.append(api)
            reasons.append(f"{api}: selected_function_line_coverage_percent={percent:g} < {args.threshold:g}")
        elif args.include_unavailable and unavailable:
            selected.append(api)
            reasons.append(f"{api}: selected_function_line_coverage_unavailable")

    selected = list(dict.fromkeys(selected))
    if args.max_apis and args.max_apis > 0:
        selected = selected[: args.max_apis]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, "".join(f"{api}\n" for api in selected))
    reason_path = out_path.with_suffix(".reasons.txt")
    atomic_write_text(reason_path, "".join(f"{line}\n" for line in reasons if line.split(":", 1)[0] in set(selected)))
    print(f"[lowcov] threshold={args.threshold:g} selected={len(selected)} from {stage4_dir / 'coverage_report.json'}")
    print(f"[lowcov] api_list={out_path}")
    print(f"[lowcov] reasons={reason_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
