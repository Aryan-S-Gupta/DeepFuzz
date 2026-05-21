#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_api_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_ok_apis(lib: str, init_dir: Path) -> Iterable[str]:
    ok = init_dir / "ok.csv"
    if not ok.exists():
        return []
    with ok.open("r", encoding="utf-8", newline="") as f:
        return [str(row.get("api_full_name", "")).strip() for row in csv.DictReader(f) if str(row.get("api_full_name", "")).strip()]


def _generic_test(api: str, init_path: Path, seed: int) -> str:
    func = _safe(api)
    return f'''from pathlib import Path

from stage4.stage4_worker import fuzz_one


def test_{func}_base_valid():
    bundle = fuzz_one(str(Path({str(init_path)!r})), mutation_budget=0, seed={seed}, case_timeout_sec=30)
    assert not bundle.get("materialization_errors"), bundle.get("materialization_errors")
    assert bundle.get("base_execution", {{}}).get("success"), bundle.get("base_execution", {{}}).get("error")
'''


def _broadcast_to_test() -> str:
    return '''import pytest
import tensorflow as tf


def test_tensorflow_broadcast_to_base_valid():
    x = tf.constant([1.0, 1.0], dtype=tf.float32)
    y = tf.broadcast_to(x, [2, 2])
    assert tuple(y.shape) == (2, 2)
    assert y.dtype == tf.float32


def test_tensorflow_broadcast_to_shape_scalar_expected_negative():
    x = tf.constant([1.0, 1.0], dtype=tf.float32)
    with pytest.raises((TypeError, ValueError, tf.errors.InvalidArgumentError)):
        tf.broadcast_to(x, 1)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="Export thesis-readable pytest tests from Stage 3 init objects")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--api-list", default="")
    ap.add_argument("--init-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    init_dir = Path(args.init_dir) if args.init_dir else ROOT / "json2init" / "results" / args.lib
    if not init_dir.is_absolute():
        init_dir = ROOT / init_dir
    if args.api_list:
        api_list = Path(args.api_list)
        if not api_list.is_absolute():
            api_list = ROOT / api_list
        apis = _read_api_list(api_list)
    else:
        apis = list(_iter_ok_apis(args.lib, init_dir))

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for api in apis:
        init_path = init_dir / f"{api}.init.json"
        init_obj = _load_json(init_path)
        if not init_path.exists() or not init_obj.get("ready_for_stage4", False):
            continue
        test_path = out_dir / f"test_{_safe(api)}.py"
        if api == "tensorflow.broadcast_to":
            body = _broadcast_to_test()
        else:
            body = _generic_test(api, init_path.resolve(), args.seed)
        test_path.write_text(body, encoding="utf-8")
        written += 1

    print(f"[tests] wrote {written} pytest files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
