from __future__ import annotations

"""Deterministic result-file helpers shared by pipeline stages."""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


RESULT_STAGE_DIRS = ("doc2info", "info2json", "json_validator", "json2init")


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def ordered_fieldnames(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    fields: List[str] = []
    for field in preferred:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for key in row.keys():
            skey = str(key)
            if skey not in fields:
                fields.append(skey)
    return fields


def atomic_write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: clean_cell(row.get(k)) for k in fields})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return []
    with target.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def merge_rows_by_api(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    selected_apis: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    selected_apis = selected_apis or set()
    merged: Dict[str, Dict[str, Any]] = {}
    for row in old_rows:
        api = clean_cell(row.get("api_full_name")).strip()
        if not api:
            continue
        if selected_apis and api in selected_apis:
            continue
        merged[api] = dict(row)
    for row in new_rows:
        api = clean_cell(row.get("api_full_name")).strip()
        if api:
            merged[api] = dict(row)
    return [merged[key] for key in sorted(merged)]


def discover_libraries(root: str | Path) -> List[str]:
    """Discover libraries from existing result directories without hardcoding."""

    base = Path(root)
    names: Set[str] = set()
    for stage in RESULT_STAGE_DIRS:
        results_dir = base / stage / "results"
        if not results_dir.exists():
            continue
        for child in results_dir.iterdir():
            if child.is_dir() and not child.name.startswith(".") and child.name != "_stale":
                names.add(child.name)
    stage4_root = base / "stage4" / "results"
    if stage4_root.exists():
        for child in stage4_root.iterdir():
            if child.is_dir() and child.name.endswith("-coverage"):
                names.add(child.name[: -len("-coverage")])
    return sorted(names)


def write_api_list(path: str | Path, apis: Iterable[str]) -> List[str]:
    names = sorted({clean_cell(api).strip() for api in apis if clean_cell(api).strip()})
    atomic_write_text(path, "".join(f"{api}\n" for api in names))
    return names
