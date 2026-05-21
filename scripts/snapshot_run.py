#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _stage4_files(stage4_dir: Path) -> Iterable[str]:
    yield from [
        "summary.json",
        "results.csv",
        "execution_summary.csv",
        "failures.csv",
        "failure.csv",
        "low_coverage.csv",
        "coverage_report.json",
        "coverage_report.md",
        "coverage_report.csv",
        "selected_function_coverage.json",
        "bug_report.json",
        "bug_report.csv",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot lightweight thesis reports before/after reruns")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage4-results-dir", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    run_dir = ROOT / "pipeline_runs" / args.lib / args.run_id
    stage4_dir = Path(args.stage4_results_dir) if args.stage4_results_dir else ROOT / "stage4" / "results" / f"{args.lib}-coverage"
    if not stage4_dir.is_absolute():
        stage4_dir = ROOT / stage4_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (args.label or "snapshot"))
    out = run_dir / "snapshots" / f"{stamp}-{label}"
    out.mkdir(parents=True, exist_ok=True)

    for rel in [
        "api_list.txt",
        "run_manifest.json",
        "final_report.json",
        "final_report.md",
        "coverage/selected_function_coverage.json",
    ]:
        _copy_if_exists(run_dir / rel, out / "pipeline_run" / rel)
    _copy_if_exists(run_dir / "triage", out / "pipeline_run" / "triage")

    for rel in _stage4_files(stage4_dir):
        _copy_if_exists(stage4_dir / rel, out / "stage4" / rel)

    manifest = {
        "library": args.lib,
        "run_id": args.run_id,
        "label": args.label or "snapshot",
        "snapshot_dir": str(out),
        "stage4_results_dir": str(stage4_dir),
    }
    (out / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[snapshot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
