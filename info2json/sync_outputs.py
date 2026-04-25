from __future__ import annotations
import csv
import re
import shutil
from pathlib import Path

def sanitize_filename(api_full_name: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', api_full_name.strip())}.json"

def sync(selected_csv: str, outdir: str) -> None:
    wanted = set()
    with open(selected_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            api = (row.get("api_full_name") or "").strip()
            if api:
                wanted.add(sanitize_filename(api))

    out_path = Path(outdir)
    stale_dir = out_path / "_stale"
    stale_dir.mkdir(parents=True, exist_ok=True)

    for p in out_path.glob("*.json"):
        if p.name in {"summary.json", "combined.json"}:
            continue
        if p.name not in wanted:
            shutil.move(str(p), str(stale_dir / p.name))
            print(f"[stale->moved] {p.name}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="selected.csv")
    ap.add_argument("--outdir", required=True, help="info2json output dir")
    args = ap.parse_args()
    sync(args.input, args.outdir)