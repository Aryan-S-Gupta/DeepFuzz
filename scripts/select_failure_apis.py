#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.result_io import atomic_write_text  # noqa: E402


AUDIT_ONLY = {
    "expected_negative_rejection",
    "negative_accepted_not_bug",
    "negative_mutation_accepted",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_api_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def iter_failure_rows(stage4_dir: Path) -> Iterable[Dict[str, str]]:
    for name in ("failures.csv", "failure.csv"):
        for row in read_csv_rows(stage4_dir / name):
            row = dict(row)
            row["_source_csv"] = name
            yield row


def main() -> int:
    ap = argparse.ArgumentParser(description="Select Stage 4 APIs with pipeline/actionable failures for merge reruns")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "latest"))
    ap.add_argument("--stage4-results-dir", default="")
    ap.add_argument("--api-list", default="")
    ap.add_argument("--include-audit", action="store_true", help="Also include negative-input audit rows")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = ROOT / "pipeline_runs" / args.lib / args.run_id
    stage4_dir = Path(args.stage4_results_dir) if args.stage4_results_dir else ROOT / "stage4" / "results" / f"{args.lib}-thesis-all-coverage"
    if not stage4_dir.is_absolute():
        stage4_dir = ROOT / stage4_dir
    api_list_path = Path(args.api_list) if args.api_list else run_dir / "api_list.txt"
    if not api_list_path.is_absolute():
        api_list_path = ROOT / api_list_path
    out_path = Path(args.out) if args.out else run_dir / "repair" / "failure_api_list.txt"
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    allowed = read_api_list(api_list_path)
    selected: List[str] = []
    reasons: List[str] = []
    for row in iter_failure_rows(stage4_dir):
        api = str(row.get("api_full_name") or row.get("api") or "").strip()
        if not api:
            continue
        if allowed and api not in allowed:
            continue
        classification = str(row.get("classification") or row.get("error_type") or "").strip()
        if not args.include_audit and classification in AUDIT_ONLY:
            continue
        source = str(row.get("_source_csv", ""))
        stage = str(row.get("stage", ""))
        rule = str(row.get("rule", ""))
        selected.append(api)
        reasons.append(f"{api}: {source}:{stage}:{classification}:{rule}")

    selected = list(dict.fromkeys(selected))
    selected_set = set(selected)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, "".join(f"{api}\n" for api in selected))
    reason_path = out_path.with_suffix(".reasons.txt")
    atomic_write_text(reason_path, "".join(f"{line}\n" for line in reasons if line.split(":", 1)[0] in selected_set))
    print(f"[failures] selected={len(selected)} from {stage4_dir}")
    print(f"[failures] api_list={out_path}")
    print(f"[failures] reasons={reason_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
