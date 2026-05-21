from __future__ import annotations

"""Shared schema contract for the DeepFuzz document pipeline.

The public API handoff is intentionally tiny: each accepted API record contains
only the qualified API name and the raw documentation text needed by downstream
generation.  Older richer CSVs remain readable, but new collection/filtering
outputs must keep exactly the two canonical columns.
"""

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


API_FULL_NAME = "api_full_name"
API_DOC_TEXT = "api_doc_text"
SIGNATURE = "signature"
KIND = "kind"
MODULE = "module"
CANONICAL_TARGET = "canonical_target"

SCHEMA_VERSION = "2.0"
STATUS_VALID = "valid"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_REPAIRED = "repaired"

CANONICAL_API_FIELDS = [API_FULL_NAME, API_DOC_TEXT]

REJECTED_FIELDS = [API_FULL_NAME, "reason", API_DOC_TEXT]

_API_ALIASES = (API_FULL_NAME, "api", "qualname", "full_name", "name")
_DOC_ALIASES = (API_DOC_TEXT, "doc", "doc_text", "documentation", "api_doc")
_EMBEDDED_SIGNATURE_PREFIX = "Observed signature: "

SUMMARY_COUNTERS = [
    "total_seen",
    "total_valid",
    "total_failed",
    "total_retried",
    "total_repaired",
    "total_unresolved",
    "total_skipped_existing_valid",
]


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def normalize_api_row(row: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a CSV/JSON row into the internal API contract.

    New writes expose only ``api_full_name`` and ``api_doc_text``.  Internal
    readers still fill compatibility metadata from older CSVs or embedded doc
    preambles so existing stage code can keep using ``SIGNATURE`` when present.
    """

    out: Dict[str, str] = {str(k): clean_cell(v) for k, v in dict(row or {}).items()}

    api = ""
    for key in _API_ALIASES:
        if clean_cell(row.get(key)).strip():
            api = clean_cell(row.get(key)).strip()
            break

    doc = ""
    for key in _DOC_ALIASES:
        if clean_cell(row.get(key)):
            doc = clean_cell(row.get(key))
            break

    signature = clean_cell(row.get(SIGNATURE)).strip() or signature_from_doc(doc)

    out[API_FULL_NAME] = api
    out[API_DOC_TEXT] = doc
    out[SIGNATURE] = signature
    for key in [KIND, MODULE, CANONICAL_TARGET]:
        out[key] = clean_cell(row.get(key))
    return out


def signature_from_doc(doc: str) -> str:
    for line in str(doc or "").splitlines()[:5]:
        stripped = line.strip()
        if stripped.lower().startswith(_EMBEDDED_SIGNATURE_PREFIX.lower()):
            return stripped.split(":", 1)[1].strip()
    return ""


def compact_doc_text(doc: str, signature: str = "") -> str:
    """Return the single downstream doc field, optionally carrying signature."""

    doc = clean_cell(doc).strip()
    signature = clean_cell(signature).strip()
    if not signature:
        return doc
    if signature_from_doc(doc):
        return doc
    return f"{_EMBEDDED_SIGNATURE_PREFIX}{signature}\n\n{doc}".strip()


def collector_entry_to_api_row(entry: Dict[str, Any]) -> Dict[str, str]:
    """Build an accepted/rejected CSV row from a collector JSONL entry."""

    return normalize_api_row(
        {
            API_FULL_NAME: entry.get("api") or entry.get("qualname") or "",
            API_DOC_TEXT: compact_doc_text(entry.get("doc") or "", entry.get("signature") or ""),
            SIGNATURE: entry.get("signature") or "",
            KIND: entry.get("kind") or "",
            MODULE: entry.get("module") or "",
            CANONICAL_TARGET: entry.get("canonical_target") or "",
        }
    )


def read_api_records(input_path: str) -> List[Dict[str, str]]:
    """Read CSV/XLSX API records with backward-compatible normalization."""

    suffix = Path(input_path).suffix.lower()
    rows: List[Dict[str, Any]]
    if suffix == ".csv":
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix in {".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError("pandas/openpyxl is required for Excel input")
        rows = pd.read_excel(input_path).fillna("").to_dict("records")
    else:
        raise ValueError(f"Unsupported API table file: {input_path}")
    return [r for r in (normalize_api_row(row) for row in rows) if r.get(API_FULL_NAME)]


def _ordered_fieldnames(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    fields: List[str] = []
    for field in preferred:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    return fields


def write_rows_csv(path: str, rows: Sequence[Dict[str, Any]], preferred_fields: Optional[Sequence[str]] = None) -> None:
    """Write rows using stable field order while preserving extra columns."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    preferred = list(preferred_fields or [])
    fieldnames = _ordered_fieldnames(list(rows), preferred)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_cell(row.get(k)) for k in fieldnames})


def write_api_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    normalized = [normalize_api_row(dict(row)) for row in rows]
    minimal = [
        {
            API_FULL_NAME: row.get(API_FULL_NAME, ""),
            API_DOC_TEXT: compact_doc_text(row.get(API_DOC_TEXT, ""), row.get(SIGNATURE, "")),
        }
        for row in normalized
    ]
    write_rows_csv(path, minimal, CANONICAL_API_FIELDS)


def write_rejected_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        n = normalize_api_row(dict(row))
        n = {API_FULL_NAME: n.get(API_FULL_NAME, ""), "reason": clean_cell(row.get("reason")), API_DOC_TEXT: n.get(API_DOC_TEXT, "")}
        normalized.append(n)
    write_rows_csv(path, normalized, REJECTED_FIELDS)


def signature_from_row(row: Dict[str, Any], fallback_doc_parser: Optional[Any] = None) -> str:
    sig = clean_cell(row.get(SIGNATURE)).strip()
    if sig:
        return sig
    if fallback_doc_parser is None:
        return ""
    return clean_cell(fallback_doc_parser(clean_cell(row.get(API_DOC_TEXT)))).strip()


def iter_api_tuples(input_path: str) -> Iterable[tuple[str, str, str]]:
    for row in read_api_records(input_path):
        yield row[API_FULL_NAME], row[API_DOC_TEXT], row.get(SIGNATURE, "")


def blank_summary() -> Dict[str, int]:
    return {key: 0 for key in SUMMARY_COUNTERS}


def summarize_stage_items(items: Sequence[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = blank_summary()
    for item in items:
        status = clean_cell(item.get("status"))
        summary["total_seen"] += 1
        if status in {STATUS_VALID, "pass", "pass_with_warnings", "ready_for_stage4", "ready"}:
            summary["total_valid"] += 1
        elif status == STATUS_REPAIRED:
            summary["total_valid"] += 1
            summary["total_repaired"] += 1
        elif status == STATUS_SKIPPED:
            summary["total_skipped_existing_valid"] += 1
        else:
            summary["total_failed"] += 1
            summary["total_unresolved"] += 1
    if extra:
        summary.update(extra)
    return summary


def stage_output(stage: str, items: Sequence[Dict[str, Any]], summary_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "items": list(items),
        "summary": summarize_stage_items(items, summary_extra),
    }


def error_signature(stage: str, api: str, error_type: str, message: str) -> str:
    key = "|".join([clean_cell(stage), clean_cell(api), clean_cell(error_type), clean_cell(message)[:1000]])
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]


def error_record(api: str, stage: str, error_type: str, message: str, payload_ref: str = "") -> Dict[str, str]:
    return {
        "api": clean_cell(api),
        "stage": clean_cell(stage),
        "error_type": clean_cell(error_type),
        "message": clean_cell(message),
        "signature": error_signature(stage, api, error_type, message),
        "payload_ref": clean_cell(payload_ref),
    }


def api_collection_json(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "items": [
            {"api": clean_cell(normalize_api_row(dict(row)).get(API_FULL_NAME)), "doc": clean_cell(normalize_api_row(dict(row)).get(API_DOC_TEXT))}
            for row in rows
            if clean_cell(normalize_api_row(dict(row)).get(API_FULL_NAME))
        ],
    }
