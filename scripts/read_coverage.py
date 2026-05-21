#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _fmt(value: Any) -> str:
    return "unavailable" if value is None or value == "" else str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print the thesis coverage numbers from a Stage 4 results directory")
    ap.add_argument("--stage4-results-dir", required=True)
    args = ap.parse_args()
    path = Path(args.stage4_results_dir) / "coverage_report.json"
    if not path.exists():
        raise SystemExit(f"coverage report not found: {path}")
    report: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    print(f"Coverage report: {path}")
    print(f"Target: {_fmt(report.get('target_library'))} {_fmt(report.get('target_version'))}")
    print(f"Main metric: {_fmt(report.get('main_coverage_number'))} ({_fmt(report.get('main_coverage_label'))})")
    print(f"API execution coverage: {_fmt(report.get('api_execution_coverage_percent'))}%")
    print(f"Selected-function coverage: {_fmt(report.get('function_line_coverage_percent'))}%")
    print(f"Package Python coverage: {_fmt(report.get('python_line_coverage_percent'))}%")
    print(f"Native coverage: {_fmt(report.get('native_line_coverage_percent'))}%")
    print(f"Valid programs: {_fmt(report.get('total_valid_programs'))} total, {_fmt(report.get('unique_valid_programs'))} unique")
    print(f"Execution time: {_fmt(report.get('execution_time_seconds'))} seconds")
    limitations = report.get("coverage_limitations") or report.get("limitations") or []
    if limitations:
        print("Limitations:")
        for item in limitations:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
