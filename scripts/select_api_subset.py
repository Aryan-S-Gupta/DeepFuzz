#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api_policy import is_internal_api  # noqa: E402
from common.result_io import atomic_write_text  # noqa: E402


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Stage 3 ok.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze a deterministic thesis API list from Stage 3 ready APIs")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Maximum APIs to write. Omit or pass 0 with --all to write every ready API.")
    ap.add_argument("--all", action="store_true", help="Write every ready API after filtering instead of truncating to --limit.")
    ap.add_argument("--public-only", action="store_true")
    ap.add_argument("--ok-csv", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if not args.all and args.limit == 0:
        raise SystemExit("pass --all to write every ready API, or pass --limit N")

    ok_csv = Path(args.ok_csv) if args.ok_csv else ROOT / "json2init" / "results" / args.lib / "ok.csv"
    rows = _read_rows(ok_csv)
    selected: List[str] = []
    seen = set()
    for row in rows:
        api = str(row.get("api_full_name", "") or row.get("api", "") or "").strip()
        if not api or api in seen:
            continue
        if args.public_only and is_internal_api(api, args.lib):
            continue
        status = str(row.get("stage3_status", "") or "").strip()
        if status and status not in {"ready_for_stage4", "spec_ready_no_smoke"}:
            continue
        selected.append(api)
        seen.add(api)
        if not args.all and len(selected) >= args.limit:
            break

    if not args.all and len(selected) < args.limit:
        print(f"[select] warning: requested {args.limit} APIs, selected {len(selected)} from {ok_csv}", file=sys.stderr)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    atomic_write_text(out, "".join(f"{api}\n" for api in selected))
    print(f"[select] wrote {len(selected)} APIs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
