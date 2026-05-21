#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
for candidate in [Path.cwd(), ROOT]:
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)

from common.pipeline_contract import (
    API_DOC_TEXT,
    API_FULL_NAME,
    SIGNATURE,
    read_api_records,
)
from common.model_config import check_model_backend, load_model_config
from common.result_io import atomic_write_csv, atomic_write_json, merge_rows_by_api

try:
    from json2init.deepfuzz_common import build_runtime_object_from_spec, import_api
except Exception:
    try:
        from deepfuzz_common import build_runtime_object_from_spec, import_api
    except Exception:
        build_runtime_object_from_spec = None
        import_api = None

SECTION_ARGS = {"args", "arguments", "parameters", "parameter", "inputs", "input"}
SECTION_RETURNS = {"returns", "return", "output", "outputs", "result", "results"}
STOP_SECTIONS = {"raises", "examples", "example", "note", "notes", "warning", "warnings", "see also", "references"}

BAD_PARAM_NAMES = {
    "optional", "required", "parameter", "parameters", "argument", "arguments", "arg", "args",
    "default", "defaults", "note", "notes", "example", "examples", "return", "returns",
    "result", "results", "dtype", "tensor", "tensors", "list", "tuple", "type", "if", "then", "else",
    "caution", "warning", "grads", "types", "matrix", "scatter", "updated",
}

COMMON_REAL_PARAM_NAMES = {
    "input", "inputs", "target", "output", "weight", "bias", "value", "device",
    "device_index", "element", "x", "y", "axis", "axes", "dim", "dims",
    "index", "indices", "shape", "size", "dtype",
}

TYPE_CANON = [
    (r"\b(?:str|string|bytes)\b", "string"),
    (r"\b(?:int|integer|long|short)\b", "int"),
    (r"\b(?:float|double|real|half|bfloat16|float16|float32|float64)\b", "float"),
    (r"\bbool(?:ean)?\b", "bool"),
    (r"\btuple\b", "tuple"),
    (r"\blist\b", "list"),
    (r"\bsequence\b", "sequence"),
    (r"\biterable\b", "sequence"),
    (r"\b(?:dict|dictionary|mapping|ordereddict)\b", "dict"),
    (r"\b(?:callable|function|fn)\b", "callable"),
    (r"\b(?:number|numeric|scalar)\b", "number"),
    (r"\b(?:future|futures)\b", "future"),
    (r"\b(?:tensor|tensors|ndarray|array|bytetensor|tensorlike)\b", "tensor"),
    (r"\bdevice\b", "device"),
    (r"\bdtype\b", "dtype"),
    (r"\bany\b", "any"),
]

DTYPE_NAMES = (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16", "bool", "complex64", "complex128",
)

STRUCTURAL_REQUIRED_PARAM_KEYS = ["type", "size", "default", "flag", "description"]
VALID_FLAGS = {"Required", "Optional", ""}
PASS_STATUSES = {"pass", "repaired", "pass_with_warnings", "zero_arg_valid_api"}
REPAIRABLE_STATUSES = {"retry", "all_params_removed_due_to_prose", "parser_failure", "needs_llm_or_doc_signature"}

REPAIR_PROMPT = """You repair exactly one API JSON spec for a documentation-grounded validator.

Return ONLY valid JSON.
No markdown.
No prose.
No comments.

API Full Name: {api_full_name}

Observed Signature Line:
{sig_line}

Deterministically Supported Parameter Names:
{supported_params}

Current Validation Issues:
{failure_summary}

Cross-Stage Error Context:
{external_failure_context}

Documentation-Derived Hints:
{doc_hints}

API Documentation Text:
\"\"\"
{api_doc_text}
\"\"\"

Current Broken JSON Spec:
{current_json}

Return exactly one JSON object with this schema:
{{
  "api_name": "",
  "module_path": "",
  "params": {{
    "<param_name>": {{
      "type": "",
      "size": "",
      "default": "",
      "flag": "",
      "description": "",
      "dtype_candidates": [],
      "enum_values": [],
      "constraints": []
    }}
  }},
  "output": {{
    "type": "",
    "numbers": "",
    "description": ""
  }},
  "constraints": []
}}

Hard rules:
1. Use only information explicitly stated in the docs or signature.
2. Only include parameters from the supported list above. If the list is empty, use only clearly documented parameters.
3. Remove fake parameters extracted from prose.
4. If a real parameter is clearly present in docs/signature, include it.
5. If uncertain, leave fields empty instead of guessing.
6. flag must be exactly one of: Required, Optional, "".
7. Keep every required schema key present.
8. Put single-parameter rules under that parameter.
9. Put only cross-parameter rules at top-level.
10. Do not invent executable defaults.
11. Keep descriptions short and literal.
12. Prefer simpler, Stage-3-friendly types such as int, float, bool, string, list, tuple, tensor, dict, sequence, dtype, device when docs allow that wording.
13. Infer whether the evidence points to Stage 1 spec extraction, Stage 2 schema validation, Stage 3 seed generation, Stage 4 mutation/coverage, or a true runtime/library issue.
14. Only repair this JSON spec when the supplied docs/signature/current JSON/error context ground the change.
15. If the cross-stage context is an environment/runtime-only issue and the docs do not justify a spec change, return the current JSON normalized to this schema without inventing API semantics.
16. Output exactly one JSON object.
"""


@dataclass
class ValidationResult:
    status: str
    classification: str
    errors: List[str]
    warnings: List[str]
    stage3_preview_ready: bool
    stage3_preview_reasons: List[str]
    param_count: int
    required_param_count: int
    supported_param_count: int
    signature_found: bool


PROSE_FIELD_PATTERNS = [
    r"\ba name for\b",
    r"\bif true\b",
    r"\bif false\b",
    r"\bwhether\b",
    r"\bdefaults?\s+to\b",
    r"\bthe direction\b",
    r"\bthe dimension\b",
    r"\bon cpu\b",
    r"\bused to\b",
]


def looks_like_prose(value: Any) -> bool:
    s = clean_text(value)
    if not s:
        return False
    low = s.lower()
    if any(re.search(pat, low) for pat in PROSE_FIELD_PATTERNS):
        return True
    words = re.findall(r"[A-Za-z]+", s)
    return len(words) > 8 and not any(ch in s for ch in "[]()|,")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def clean_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if not isinstance(v, str):
        return ""
    s = normalize_ws(v)
    return "" if s.lower() in {"null", "none", "nan"} else s


def split_lines(doc: str) -> List[str]:
    return str(doc or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def is_suspicious_param_name(name: str, allowed_names: Optional[Sequence[str]] = None) -> bool:
    n = normalize_ws(name).lower().strip("*` ")
    if not n:
        return True
    if n in {"self", "cls"}:
        return True
    if allowed_names and n in {x.lower() for x in allowed_names}:
        return False
    if n in COMMON_REAL_PARAM_NAMES:
        return False
    if n in BAD_PARAM_NAMES:
        return True
    if len(n) > 64:
        return True
    return False


def read_api_docs(input_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_api_records(input_path):
        api = str(row.get(API_FULL_NAME, "") or "").strip()
        if api:
            out[api] = row
    return out


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def write_json(path: str, data: Any) -> None:
    atomic_write_json(path, data)


def write_table(base_path_without_ext: str, rows: List[Dict[str, Any]]) -> None:
    csv_path = base_path_without_ext + ".csv"
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    atomic_write_csv(csv_path, rows, fields)


def load_table(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except Exception:
        return default


def merge_rows_by_api_legacy(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    replace_api_names: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    replace_api_names = replace_api_names or set()

    for row in old_rows:
        api = str(row.get("api_full_name", "") or "").strip()
        if not api or api in replace_api_names:
            continue
        merged[api] = row

    for row in new_rows:
        api = str(row.get("api_full_name", "") or "").strip()
        if not api:
            continue
        merged[api] = row

    return [merged[k] for k in sorted(merged)]


def find_signature_line(doc: str) -> str:
    lines = [ln.strip() for ln in split_lines(doc) if ln.strip()]
    sig_pat = re.compile(r"^[A-Za-z_][\w.]*\([^)]*\)(?:\s*->\s*[^:]+)?$")
    for line in lines[:30]:
        if sig_pat.match(line):
            return line
    for line in lines:
        if sig_pat.match(line):
            return line
    return ""


def parse_signature_params(sig_line: str) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
    if not sig_line or "(" not in sig_line or ")" not in sig_line:
        return [], {}, None
    try:
        params_part = sig_line[sig_line.find("(") + 1:sig_line.rfind(")")]
        fake = f"def _f({params_part}):\n    pass"
        fn = ast.parse(fake).body[0]
        assert isinstance(fn, ast.FunctionDef)

        params: List[str] = []
        defaults: Dict[str, Any] = {}
        all_pos = list(fn.args.posonlyargs) + list(fn.args.args)

        for arg in all_pos:
            if not is_suspicious_param_name(arg.arg):
                params.append(arg.arg)

        if fn.args.defaults:
            start = len(all_pos) - len(fn.args.defaults)
            for idx, node in enumerate(fn.args.defaults):
                name = all_pos[start + idx].arg
                if is_suspicious_param_name(name):
                    continue
                try:
                    defaults[name] = ast.literal_eval(node)
                except Exception:
                    defaults[name] = ast.unparse(node) if hasattr(ast, "unparse") else ""

        for arg, node in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if is_suspicious_param_name(arg.arg):
                continue
            params.append(arg.arg)
            if node is not None:
                try:
                    defaults[arg.arg] = ast.literal_eval(node)
                except Exception:
                    defaults[arg.arg] = ast.unparse(node) if hasattr(ast, "unparse") else ""

        ret = normalize_ws(sig_line.split("->", 1)[1]) if "->" in sig_line else None
        return params, defaults, ret
    except Exception:
        return [], {}, None


def extract_default_from_text(text: str) -> str:
    s = clean_text(text)
    if not s:
        return ""
    patterns = [
        r"\bDefaults to\s+([^.]+)",
        r"\bdefault(?:s)?\s+is\s+([^.]+)",
        r"\bdefault\s*:\s*([^.]+)",
        r"\bis\s+``([^`]+)``\s+\(default\)",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return normalize_ws(m.group(1)).strip(",;: ").strip("`")
    return ""


def explicit_optional_from_text(text: str) -> bool:
    d = clean_text(text).lower()
    return bool(
        d
        and (
            "(optional)" in d
            or d.startswith("optional")
            or " optional" in d
            or " defaults to " in d
            or d.startswith("defaults to ")
            or " default is " in d
            or " default: " in d
            or re.search(r"\bdefault\b", d)
            or "if none" in d
        )
    )


def extract_size_text(text: str) -> str:
    s = normalize_ws(text)
    if not s:
        return ""
    patterns = [
        r"\b\d+\s*[- ]?d\b",
        r"\b\d+\s*[- ]?d(?:imension|im)?s?\b",
        r"\bshape\s*=\s*\([^)]*\)",
        r"\bshape\s+\([^)]*\)",
        r"\btuple of length \d+\b",
        r"\blist of length \d+\b",
        r"\bscalar\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return normalize_ws(m.group(0)).replace("-d", "-D")
    return ""


def normalize_type(type_str: str, desc_text: str = "") -> str:
    text = " ".join(x for x in [clean_text(type_str), clean_text(desc_text)] if x)
    if not text:
        return ""
    if looks_like_prose(type_str):
        text = clean_text(desc_text)
    out: List[str] = []
    seen = set()
    low = text.lower()
    if re.search(r"\bByteTensor\b", text):
        return "tensor"
    for pat, label in TYPE_CANON:
        if re.search(pat, low):
            if label not in seen:
                seen.add(label)
                out.append(label)
    return "|".join(out) if out else clean_text(type_str)


def cleanup_machine_field(value: Any, field: str) -> str:
    s = clean_text(value)
    if not s:
        return ""
    if looks_like_prose(s):
        return ""
    if field == "size" and len(s.split()) > 4 and not re.search(r"[\[\]()]|\b\d+\s*[- ]?d\b", s, re.I):
        return ""
    if field == "default":
        s = s.strip("` ")
        if re.match(r"^(?:None|null|True|False|true|false|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\[\]|\{\}|\(\)|['\"][^'\"]{0,80}['\"]|(?:tf|tensorflow|torch|np|numpy)\.[A-Za-z_][\w.]*)$", s, re.I):
            return s
        literal = re.match(r"^(?:[-+]?\d+(?:\.\d+)?|True|False|None|null|true|false)\b", s)
        return literal.group(0) if literal else ""
    return s


def runtime_signature_info(api_full_name: str) -> Tuple[List[str], bool, bool]:
    if import_api is None:
        return [], False, False
    try:
        obj = import_api(api_full_name)
        sig = inspect.signature(obj)
    except Exception:
        return [], False, False
    params: List[str] = []
    accepts_kwargs = False
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if not is_suspicious_param_name(name):
            params.append(name)
    return params, accepts_kwargs, True


def classify_empty_params(
    api_full_name: str,
    supported_names: Sequence[str],
    sig_line: str,
    original_spec: Optional[Dict[str, Any]] = None,
) -> str:
    original_params = original_spec.get("params", {}) if isinstance(original_spec, dict) and isinstance(original_spec.get("params"), dict) else {}
    if original_params and all(is_suspicious_param_name(str(name), supported_names if supported_names else None) or looks_like_prose(str(name)) for name in original_params):
        return "all_params_removed_due_to_prose"
    sig_params, _sig_defaults, _sig_ret = parse_signature_params(sig_line)
    runtime_params, _accepts_kwargs, runtime_signature_found = runtime_signature_info(api_full_name)
    if sig_line and not sig_params:
        return "zero_arg_valid_api"
    if runtime_signature_found and not runtime_params:
        return "zero_arg_valid_api"
    if not sig_line and not runtime_signature_found:
        return "needs_llm_or_doc_signature"
    if not supported_names and (sig_params or runtime_params):
        return "parser_failure"
    return "no_signature_available"


def parse_doc_params(
    doc: str,
    allowed_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, bool], Dict[str, str]]:
    lines = split_lines(doc)
    type_map: Dict[str, str] = {}
    desc_map: Dict[str, str] = {}
    size_map: Dict[str, str] = {}
    optional_map: Dict[str, bool] = {}
    default_map: Dict[str, str] = {}
    allowed_lookup = {x.lower(): x for x in (allowed_names or [])}

    def accept_name(name: str) -> Optional[str]:
        name = name.lstrip("*")
        if allowed_lookup:
            return allowed_lookup.get(name.lower())
        if is_suspicious_param_name(name):
            return None
        return name

    in_args = False
    current_name: Optional[str] = None
    p_google = re.compile(r"^\s*(\*{0,2}[A-Za-z_][\w\.]*)\s*(\(([^)]*)\))?\s*:\s*(.*)$")
    p_numpy = re.compile(r"^\s*(\*{0,2}[A-Za-z_][\w\.]*)\s*:\s*(.+?)\s*$")
    p_rst_param = re.compile(r"^\s*:param\s+([A-Za-z_][\w]*)\s*:\s*(.+)$")
    p_rst_type = re.compile(r"^\s*:type\s+([A-Za-z_][\w]*)\s*:\s*(.+)$")

    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.rstrip(":").lower()
        if lower in SECTION_ARGS:
            in_args = True
            current_name = None
            continue
        if in_args and not stripped:
            current_name = None
            continue
        if in_args and lower in SECTION_RETURNS.union(STOP_SECTIONS):
            in_args = False
            current_name = None
            continue

        m = p_rst_param.match(line)
        if m:
            name = accept_name(m.group(1))
            if not name:
                continue
            desc = normalize_ws(m.group(2))
            desc_map[name] = desc
            optional_map[name] = explicit_optional_from_text(desc)
            if (size := extract_size_text(desc)):
                size_map[name] = size
            if (default := extract_default_from_text(desc)):
                default_map[name] = default
            continue

        m = p_rst_type.match(line)
        if m:
            name = accept_name(m.group(1))
            if not name:
                continue
            typ = normalize_ws(m.group(2))
            type_map[name] = typ
            optional_map[name] = optional_map.get(name, False) or explicit_optional_from_text(typ)
            if (size := extract_size_text(typ)) and name not in size_map:
                size_map[name] = size
            if (default := extract_default_from_text(typ)) and name not in default_map:
                default_map[name] = default
            continue

        if not in_args:
            continue

        if current_name and (line.startswith(" ") or line.startswith("\t")):
            m_cont = p_google.match(line)
            if m_cont:
                raw_candidate = m_cont.group(1).lstrip("*")
                accepted = accept_name(raw_candidate)
                indent = len(line) - len(line.lstrip(" \t"))
                if accepted and indent <= 4:
                    current_name = accepted
                    meta = normalize_ws(m_cont.group(3))
                    desc = normalize_ws(m_cont.group(4))
                    if meta:
                        type_map[current_name] = meta
                    if desc:
                        desc_map[current_name] = desc
                    continue

            extra = normalize_ws(line)
            if extra:
                prev = desc_map.get(current_name, "")
                desc_map[current_name] = normalize_ws((prev + " " + extra).strip())
                optional_map[current_name] = optional_map.get(current_name, False) or explicit_optional_from_text(extra)
                if (size := extract_size_text(extra)) and current_name not in size_map:
                    size_map[current_name] = size
                if (default := extract_default_from_text(extra)) and current_name not in default_map:
                    default_map[current_name] = default
            continue

        m = p_google.match(line)
        if m:
            name = accept_name(m.group(1))
            if not name:
                current_name = None
                continue
            meta = normalize_ws(m.group(3))
            desc = normalize_ws(m.group(4))
            current_name = name
            if meta:
                type_map[name] = meta
                if (size := extract_size_text(meta)):
                    size_map[name] = size
                if (default := extract_default_from_text(meta)):
                    default_map[name] = default
            if desc:
                desc_map[name] = desc
                if (size := extract_size_text(desc)):
                    size_map[name] = size
                if (default := extract_default_from_text(desc)):
                    default_map[name] = default
            optional_map[name] = explicit_optional_from_text(meta) or explicit_optional_from_text(desc)
            continue

        m = p_numpy.match(line)
        if m and idx + 1 < len(lines):
            name = accept_name(m.group(1))
            if not name:
                current_name = None
                continue
            meta = normalize_ws(m.group(2))
            current_name = name
            if meta:
                type_map[name] = meta
                if (size := extract_size_text(meta)):
                    size_map[name] = size
                if (default := extract_default_from_text(meta)):
                    default_map[name] = default
            optional_map[name] = explicit_optional_from_text(meta)

    return type_map, desc_map, size_map, optional_map, default_map


def parse_return_info(doc: str) -> Tuple[str, str, str]:
    lines = split_lines(doc)
    in_returns = False
    ret_type = ""
    ret_desc_parts: List[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.rstrip(":").lower()
        if lower in SECTION_RETURNS:
            in_returns = True
            continue
        if in_returns and not stripped:
            continue
        if in_returns and lower in SECTION_ARGS.union(STOP_SECTIONS):
            break
        if not in_returns:
            continue
        if not ret_type:
            m = re.match(r"^\s*([A-Za-z_][\w\[\], .|/-]*)\s*:\s*(.+)$", stripped)
            if m:
                ret_type = normalize_ws(m.group(1))
                ret_desc_parts.append(normalize_ws(m.group(2)))
                continue
        ret_desc_parts.append(normalize_ws(stripped))

    if not ret_desc_parts:
        first = normalize_ws(split_lines(doc)[0]) if split_lines(doc) else ""
        if first.lower().startswith("return"):
            ret_desc_parts.append(first)

    ret_desc = normalize_ws(" ".join(ret_desc_parts))
    if re.search(r"tuple of two|named 2-tuple|two values", ret_desc, flags=re.I):
        ret_numbers = "2"
    else:
        ret_numbers = "1" if ret_desc else ""

    return normalize_type(ret_type, ret_desc), ret_numbers, ret_desc


def extract_dtype_candidates(*texts: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for text in texts:
        t = str(text or "")
        for dt in DTYPE_NAMES:
            if re.search(rf"\b{re.escape(dt)}\b", t, flags=re.I):
                if dt not in seen:
                    seen.add(dt)
                    out.append(dt)
    return out


def extract_enum_values(*texts: str, param_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    param_lower = {p.lower() for p in param_names}
    for text in texts:
        s = str(text or "")
        if "one of" not in s.lower() and "either" not in s.lower() and "options" not in s.lower():
            continue
        for cand in re.findall(r"[`'\"]([A-Za-z0-9_.-]+)[`'\"]", s):
            low = cand.lower()
            if low in BAD_PARAM_NAMES or low in param_lower:
                continue
            if low not in seen:
                seen.add(low)
                out.append(cand)
    return out


def normalize_string_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        s = clean_text(str(item))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_constraint_list(raw: Any) -> List[str]:
    return normalize_string_list(raw)


def get_supported_param_names(doc: str, signature: str = "") -> Tuple[List[str], Dict[str, Any], str, Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, bool], Dict[str, str]]:
    sig_line = clean_text(signature) or find_signature_line(doc)
    sig_params, sig_defaults, _ = parse_signature_params(sig_line)
    doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map = parse_doc_params(doc, allowed_names=sig_params if sig_params else None)
    supported_names: List[str] = []
    seen = set()
    for p in sig_params + list(doc_type_map.keys()) + list(doc_desc_map.keys()):
        if is_suspicious_param_name(p):
            continue
        if p not in seen:
            seen.add(p)
            supported_names.append(p)
    return supported_names, sig_defaults, sig_line, doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map


def normalize_schema(spec: Dict[str, Any], api_full_name: str, doc: str, signature: str = "") -> Dict[str, Any]:
    module_path, _, api_name = api_full_name.rpartition(".")
    supported_names, sig_defaults, _sig_line, doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map = get_supported_param_names(doc, signature)
    ret_type_doc, ret_numbers_doc, ret_desc_doc = parse_return_info(doc)
    llm_params = spec.get("params", {}) if isinstance(spec.get("params"), dict) else {}

    if not supported_names:
        for p in llm_params.keys():
            if not is_suspicious_param_name(p):
                supported_names.append(p)

    out = {
        "api_name": api_name,
        "module_path": module_path,
        "params": {},
        "output": {
            "type": ret_type_doc or normalize_type(clean_text((spec.get("output") or {}).get("type", ""))),
            "numbers": clean_text((spec.get("output") or {}).get("numbers", "")) or ret_numbers_doc,
            "description": ret_desc_doc or clean_text((spec.get("output") or {}).get("description", "")),
        },
        "constraints": normalize_constraint_list(spec.get("constraints")),
    }

    for name in supported_names:
        lname = clean_text(name).lower()
        if api_full_name.startswith("tensorflow.") and lname == "name":
            continue
        if is_suspicious_param_name(name):
            continue
        p = llm_params.get(name, {}) if isinstance(llm_params.get(name), dict) else {}
        desc = clean_text(doc_desc_map.get(name, "")) or clean_text(p.get("description", ""))
        raw_type = clean_text(doc_type_map.get(name, "")) or clean_text(p.get("type", ""))

        if name in sig_defaults:
            default = clean_text(sig_defaults[name])
        elif name in doc_default_map:
            default = clean_text(doc_default_map[name])
        else:
            default = clean_text(p.get("default", ""))

        flag = clean_text(p.get("flag", ""))
        if name in sig_defaults or doc_optional_map.get(name, False) or flag == "Optional":
            final_flag = "Optional"
        else:
            final_flag = "Required" if supported_names else ""

        constraints = normalize_constraint_list(p.get("constraints"))

        enum_values = normalize_string_list(p.get("enum_values")) or normalize_string_list(p.get("valid_values")) or extract_enum_values(raw_type, desc, *constraints, param_names=supported_names)
        case_insensitive = bool(p.get("case_insensitive") or p.get("enum_case_insensitive"))
        if api_full_name.startswith("jax.") and lname == "padding":
            enum_values = ["VALID", "SAME", "SAME_LOWER"]
            case_insensitive = True
        out["params"][name] = {
            "type": normalize_type(raw_type, desc),
            "size": cleanup_machine_field(clean_text(doc_size_map.get(name, "")) or extract_size_text(clean_text(p.get("size", ""))), "size"),
            "default": cleanup_machine_field(default, "default"),
            "flag": final_flag,
            "description": desc,
            "dtype_candidates": normalize_string_list(p.get("dtype_candidates")) or extract_dtype_candidates(raw_type, desc, *constraints),
            "enum_values": enum_values,
            "case_insensitive": case_insensitive,
            "constraints": constraints,
        }

    return out


def doc_hints_block(api_full_name: str, doc: str, signature: str = "") -> str:
    supported_names, sig_defaults, sig_line, doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map = get_supported_param_names(doc, signature)
    rows = [
        f"signature={sig_line or '(not found)'}",
        f"supported_params={', '.join(supported_names) if supported_names else '(none)'}",
    ]
    for name in supported_names[:32]:
        rows.append(
            f"param={name}; type={doc_type_map.get(name, '')}; optional={doc_optional_map.get(name, False)}; default={doc_default_map.get(name, sig_defaults.get(name, ''))}; size={doc_size_map.get(name, '')}; desc={doc_desc_map.get(name, '')}"
        )
    return "\n".join(rows)


def call_ollama_json(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 300,
    temperature: float = 0.0,
    num_predict: int = 700,
    num_ctx: int = 4096,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for Ollama repair; install requirements.txt")
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    raw = str(data.get("response", "")).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass
    return {}


def build_prompt(api_full_name: str, doc: str, current_spec: Dict[str, Any], errors: List[str], warnings: List[str], signature: str = "") -> str:
    supported_names, _sig_defaults, sig_line, _doc_type_map, _doc_desc_map, _doc_size_map, _doc_optional_map, _doc_default_map = get_supported_param_names(doc, signature)
    return REPAIR_PROMPT.format(
        api_full_name=api_full_name,
        sig_line=sig_line or "",
        supported_params=", ".join(supported_names) if supported_names else "(none reliably parsed)",
        failure_summary=" | ".join(errors + [f"warning: {w}" for w in warnings]),
        external_failure_context="\n".join(errors[:24]),
        doc_hints=doc_hints_block(api_full_name, doc, signature),
        api_doc_text=doc,
        current_json=json.dumps(current_spec, ensure_ascii=False, indent=2),
    )


def preview_stage3_readiness(api_full_name: str, spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if build_runtime_object_from_spec is None:
        return True, []
    try:
        runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
        ready = bool(getattr(runtime_obj, "spec_ready", False))
        reasons = list(getattr(runtime_obj, "readiness_reasons", []) or [])
        return ready, reasons
    except Exception as exc:
        return False, [f"stage3 preview exception: {type(exc).__name__}: {exc}"]


def validate_normalized_spec(
    api_full_name: str,
    spec: Dict[str, Any],
    doc: str,
    signature: str = "",
    original_spec: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    module_path, _, api_name = api_full_name.rpartition(".")
    supported_names, _sig_defaults, sig_line, doc_type_map, doc_desc_map, _doc_size_map, _doc_optional_map, _doc_default_map = get_supported_param_names(doc, signature)

    if not clean_text(doc):
        errors.append("missing source doc")

    if not isinstance(spec, dict):
        errors.append("spec is not a JSON object")
        return ValidationResult("parser_failure", "parser_failure", errors, warnings, False, [], 0, 0, len(supported_names), bool(sig_line))

    if clean_text(spec.get("api_name", "")) != api_name:
        errors.append(f"api_name mismatch: expected '{api_name}'")
    if clean_text(spec.get("module_path", "")) != module_path:
        errors.append(f"module_path mismatch: expected '{module_path}'")

    params = spec.get("params")
    if not isinstance(params, dict):
        errors.append("params must be an object")
        params = {}

    output = spec.get("output")
    if not isinstance(output, dict):
        errors.append("output must be an object")
        output = {}

    empty_param_status = ""
    if not params:
        empty_param_status = classify_empty_params(api_full_name, supported_names, sig_line, original_spec=original_spec)
        if empty_param_status == "zero_arg_valid_api":
            warnings.append("zero-argument callable accepted")
        else:
            errors.append(empty_param_status)

    required_count = 0
    seen_required_doc_signal = False
    supported_lower = {x.lower() for x in supported_names}

    if supported_names:
        for name in supported_names:
            desc = clean_text(doc_desc_map.get(name, ""))
            typ = clean_text(doc_type_map.get(name, ""))
            if desc or typ:
                seen_required_doc_signal = True
                break

    for name, meta in params.items():
        if is_suspicious_param_name(name, supported_names if supported_names else None):
            errors.append(f"suspicious or hallucinated parameter name: {name}")
            continue
        if supported_names and name.lower() not in supported_lower:
            errors.append(f"parameter not supported by docs/signature: {name}")
        if not isinstance(meta, dict):
            errors.append(f"param '{name}' must be an object")
            continue
        for key in STRUCTURAL_REQUIRED_PARAM_KEYS:
            if key not in meta:
                errors.append(f"param '{name}' missing key '{key}'")
        flag = clean_text(meta.get("flag", ""))
        if flag not in VALID_FLAGS:
            errors.append(f"param '{name}' has invalid flag '{flag}'")
        if flag == "Required":
            required_count += 1
            if not clean_text(meta.get("type", "")):
                warnings.append(f"required param '{name}' missing type")
            if not clean_text(meta.get("description", "")):
                warnings.append(f"required param '{name}' missing description")
        elif flag == "":
            warnings.append(f"param '{name}' has empty flag")
        for field in ["type", "size", "default"]:
            raw = clean_text(meta.get(field, ""))
            if raw and looks_like_prose(raw):
                errors.append(f"param '{name}' {field} field contains prose")

    if supported_names and seen_required_doc_signal and required_count == 0:
        warnings.append("no required params remained after normalization; check doc parsing")

    for key in ["type", "numbers", "description"]:
        if key not in output:
            errors.append(f"output missing key '{key}'")

    if not clean_text(output.get("type", "")) and not clean_text(output.get("description", "")):
        warnings.append("output is weakly specified")

    if "constraints" not in spec or not isinstance(spec.get("constraints"), list):
        errors.append("top-level constraints must be a list")

    stage3_ready, stage3_reasons = preview_stage3_readiness(api_full_name, spec)
    if not stage3_ready:
        warnings.append("stage3 preview not ready: " + " | ".join(stage3_reasons[:3]))

    if empty_param_status == "zero_arg_valid_api" and not [e for e in errors if e != "zero_arg_valid_api"]:
        status = "zero_arg_valid_api"
        classification = "zero_arg_valid_api"
    elif empty_param_status and errors == [empty_param_status]:
        status = empty_param_status
        classification = empty_param_status
    elif errors:
        status = "retry"
        classification = "parser_failure" if any("params" in e or "parameter" in e for e in errors) else "validation_error"
    elif warnings:
        status = "pass_with_warnings"
        classification = "pass_with_warnings"
    else:
        status = "pass"
        classification = "pass"

    return ValidationResult(
        status=status,
        classification=classification,
        errors=errors,
        warnings=warnings,
        stage3_preview_ready=stage3_ready,
        stage3_preview_reasons=stage3_reasons,
        param_count=len(params),
        required_param_count=required_count,
        supported_param_count=len(supported_names),
        signature_found=bool(sig_line),
    )


def choose_model_for_round(round_idx: int, primary: str, fallback: str, fallback_after_round: int) -> str:
    if fallback and round_idx >= max(2, fallback_after_round):
        return fallback
    return primary


def load_only_api_set(path: str) -> Optional[set[str]]:
    if not path:
        return None
    names: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def load_external_errors(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not path:
        return out
    try:
        if str(path).lower().endswith(".csv"):
            for row in load_table(path):
                api = str(row.get("api") or row.get("api_full_name") or "").strip()
                if not api:
                    continue
                stage = str(row.get("stage", "") or "")
                etype = str(row.get("error_type") or row.get("status") or row.get("stage3_status") or "")
                msg = str(row.get("message") or row.get("error") or row.get("reason") or row.get("errors") or "").strip()
                if msg:
                    out.setdefault(api, []).append(f"{stage}: {etype}: {msg}")
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                api = str(row.get("api") or row.get("api_full_name") or "").strip()
                if not api:
                    continue
                stage = str(row.get("stage", "") or "")
                etype = str(row.get("error_type", "") or "")
                msg = str(row.get("message") or row.get("error") or row.get("reason") or "").strip()
                if msg:
                    out.setdefault(api, []).append(f"{stage}: {etype}: {msg}")
    except Exception:
        return out
    return out


def process(
    spec_dir: str,
    api_csv: str,
    state_dir: str,
    primary_repair_model: str = "mistral:7b",
    fallback_repair_model: str = "mixtral:8x7b",
    fallback_after_round: int = 3,
    repair_host: str = "http://localhost:11434",
    max_rounds: int = 3,
    repair_timeout: int = 300,
    repair_num_predict: int = 700,
    repair_num_ctx: int = 4096,
    repair_temperature: float = 0.0,
    only_apis: str = "",
    force: bool = False,
    external_errors_csv: str = "",
    external_errors_jsonl: str = "",
) -> None:
    docs = read_api_docs(api_csv)
    spec_root = Path(spec_dir)
    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)

    allow_only = load_only_api_set(only_apis)
    external_errors = load_external_errors(external_errors_csv or external_errors_jsonl)
    prev_ok_rows = load_table(str(state_root / "ok.csv"))
    prev_error_rows = load_table(str(state_root / "errors.csv"))
    prev_ok_by_api = {str(row.get("api_full_name", "") or ""): row for row in prev_ok_rows}
    replace_names = set(allow_only or set())

    current_rows: List[Dict[str, Any]] = []

    total_seen = 0
    selected_this_run = 0
    skipped_existing_valid = 0

    for spec_path in sorted(spec_root.glob("*.json")):
        if spec_path.name == "summary.json":
            continue

        api_full_name = spec_path.stem
        total_seen += 1

        if allow_only is not None and api_full_name not in allow_only:
            continue

        prev_row = prev_ok_by_api.get(api_full_name)
        if not force and prev_row and str(prev_row.get("status", "")) in PASS_STATUSES:
            selected_this_run += 1
            skipped_existing_valid += 1
            current_rows.append(prev_row)
            continue

        selected_this_run += 1
        current_spec = load_json(str(spec_path))
        doc_row = docs.get(api_full_name, {})
        doc = str(doc_row.get(API_DOC_TEXT, "") or "")
        signature = str(doc_row.get(SIGNATURE, "") or "")

        if not doc:
            row = {
                "api_full_name": api_full_name,
                "status": "fail",
                "classification": "missing_source_doc",
                "rounds": 0,
                "repair_model_used": "",
                "signature_found": False,
                "param_count": 0,
                "required_param_count": 0,
                "supported_param_count": 0,
                "stage3_preview_ready": False,
                "errors": "missing source doc",
                "warnings": "",
                "stage3_preview_reasons": "",
                "json_path": str(spec_path),
            }
            current_rows.append(row)
            continue

        normalized = normalize_schema(current_spec, api_full_name, doc, signature)
        result = validate_normalized_spec(api_full_name, normalized, doc, signature, original_spec=current_spec)

        final_status = result.status
        model_used = "deterministic"
        rounds_used = 1
        repaired = False
        model_health_checked = False
        model_health_ok = True
        model_health_message = ""

        if result.status in REPAIRABLE_STATUSES and primary_repair_model:
            for round_idx in range(2, max_rounds + 1):
                if not model_health_checked:
                    cfg = load_model_config()
                    model_health_ok, model_health_message = check_model_backend(primary_repair_model, repair_host, backend=cfg.backend)
                    model_health_checked = True
                if not model_health_ok:
                    final_status = "environment_model_unavailable"
                    result.errors = [model_health_message]
                    model_used = "unavailable"
                    break
                rounds_used = round_idx
                model_used = choose_model_for_round(round_idx, primary_repair_model, fallback_repair_model, fallback_after_round)
                prompt = build_prompt(api_full_name, doc, normalized, result.errors + external_errors.get(api_full_name, []), result.warnings, signature)
                candidate = call_ollama_json(
                    prompt=prompt,
                    model=model_used,
                    host=repair_host,
                    timeout=repair_timeout,
                    temperature=repair_temperature,
                    num_predict=repair_num_predict,
                    num_ctx=repair_num_ctx,
                )
                if candidate:
                    normalized = normalize_schema(candidate, api_full_name, doc, signature)
                result = validate_normalized_spec(api_full_name, normalized, doc, signature, original_spec=current_spec)
                if result.status in PASS_STATUSES:
                    repaired = True
                    final_status = "repaired" if result.warnings or round_idx > 1 else result.status
                    break
            else:
                final_status = result.status
        else:
            final_status = result.status

        if final_status in PASS_STATUSES or repaired:
            write_json(str(spec_path), normalized)

        row = {
            "api_full_name": api_full_name,
            "status": "repaired" if repaired else final_status,
            "classification": result.classification,
            "rounds": rounds_used,
            "repair_model_used": model_used,
            "signature_found": result.signature_found,
            "param_count": result.param_count,
            "required_param_count": result.required_param_count,
            "supported_param_count": result.supported_param_count,
            "stage3_preview_ready": result.stage3_preview_ready,
            "errors": " | ".join(result.errors),
            "warnings": " | ".join(result.warnings),
            "stage3_preview_reasons": " | ".join(result.stage3_preview_reasons),
            "json_path": str(spec_path),
        }
        current_rows.append(row)

    if allow_only:
        report_rows = merge_rows_by_api(prev_ok_rows + prev_error_rows, current_rows, replace_names)
    else:
        report_rows = sorted(current_rows, key=lambda r: str(r.get("api_full_name", "")))

    ok_rows = [r for r in report_rows if str(r.get("status")) in PASS_STATUSES]
    unresolved_rows = [r for r in report_rows if str(r.get("status")) not in PASS_STATUSES]

    summary = {
        "schema_version": "2.0",
        "total_seen": total_seen,
        "selected": selected_this_run,
        "merged_total_rows": len(report_rows),
        "pass": sum(1 for r in report_rows if str(r.get("status")) == "pass"),
        "pass_with_warnings": sum(1 for r in report_rows if str(r.get("status")) == "pass_with_warnings"),
        "zero_arg_valid_api": sum(1 for r in report_rows if str(r.get("status")) == "zero_arg_valid_api"),
        "needs_llm_or_doc_signature": sum(1 for r in report_rows if str(r.get("status")) == "needs_llm_or_doc_signature"),
        "no_signature_available": sum(1 for r in report_rows if str(r.get("status")) == "no_signature_available"),
        "unsupported_api_kind": sum(1 for r in report_rows if str(r.get("status")) == "unsupported_api_kind"),
        "all_params_removed_due_to_prose": sum(1 for r in report_rows if str(r.get("status")) == "all_params_removed_due_to_prose"),
        "parser_failure": sum(1 for r in report_rows if str(r.get("status")) == "parser_failure"),
        "environment_model_unavailable": sum(1 for r in report_rows if str(r.get("status")) == "environment_model_unavailable"),
        "repaired": sum(1 for r in report_rows if str(r.get("status")) == "repaired"),
        "retry": sum(1 for r in unresolved_rows if str(r.get("status")) in REPAIRABLE_STATUSES),
        "fail": sum(1 for r in unresolved_rows if str(r.get("status")) not in REPAIRABLE_STATUSES),
        "missing_doc": sum(1 for r in unresolved_rows if "missing source doc" in str(r.get("errors", "")).lower()),
        "total_valid": len(ok_rows),
        "total_failed": len(unresolved_rows),
        "total_retried": sum(1 for r in report_rows if safe_int(r.get("rounds", 0)) > 1),
        "total_repaired": sum(1 for r in report_rows if str(r.get("status")) == "repaired"),
        "total_unresolved": len(unresolved_rows),
        "total_skipped_existing_valid": skipped_existing_valid,
        "models": {
            "deterministic": "deterministic",
            "primary_repair_model": primary_repair_model,
            "fallback_repair_model": fallback_repair_model,
        },
    }

    ok_fields = [
        "api_full_name", "status", "classification", "rounds", "repair_model_used",
        "signature_found", "param_count", "required_param_count", "supported_param_count",
        "stage3_preview_ready", "errors", "warnings", "stage3_preview_reasons", "json_path",
    ]
    error_fields = list(ok_fields)
    atomic_write_csv(state_root / "ok.csv", [{k: row.get(k, "") for k in ok_fields} for row in ok_rows], ok_fields)
    atomic_write_csv(
        state_root / "errors.csv",
        [{k: row.get(k, "") for k in error_fields} for row in unresolved_rows],
        error_fields,
    )

    print(
        f"[validator] selected_this_run={summary['selected']} merged_total={summary['merged_total_rows']} "
        f"pass={summary['pass']} pass_with_warnings={summary['pass_with_warnings']} "
        f"zero_arg={summary['zero_arg_valid_api']} repaired={summary['repaired']} "
        f"retry={summary['retry']} fail={summary['fail']}"
    )
    print(f"[validator] ok: {state_root / 'ok.csv'}")
    print(f"[validator] errors: {state_root / 'errors.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 validator/repairer for documentation-grounded API JSON specs")
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--api-csv", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--primary-repair-model", default="mistral:7b")
    ap.add_argument("--repair-model", dest="primary_repair_model", default=argparse.SUPPRESS, help="Backward-compatible alias for --primary-repair-model")
    ap.add_argument("--fallback-repair-model", default="mixtral:8x7b")
    ap.add_argument("--fallback-after-round", type=int, default=3, help="Use fallback model from this repair round onward.")
    ap.add_argument("--repair-host", default="http://localhost:11434")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--repair-timeout", type=int, default=300)
    ap.add_argument("--repair-num-predict", type=int, default=700)
    ap.add_argument("--repair-num-ctx", type=int, default=4096)
    ap.add_argument("--repair-temperature", type=float, default=0.0)
    ap.add_argument("--only-apis", default="", help="Optional text file with one api_full_name per line for targeted retry.")
    ap.add_argument("--force", action="store_true", help="Revalidate APIs that were already valid in the previous run.")
    ap.add_argument("--external-errors-csv", default="", help="Optional consolidated cross-stage error context for repair prompts.")
    ap.add_argument("--external-errors-jsonl", default="", help="Backward-compatible alias; CSV is the canonical repair context.")
    args = ap.parse_args()

    process(
        spec_dir=args.spec_dir,
        api_csv=args.api_csv,
        state_dir=args.state_dir,
        primary_repair_model=args.primary_repair_model,
        fallback_repair_model=args.fallback_repair_model,
        fallback_after_round=args.fallback_after_round,
        repair_host=args.repair_host,
        max_rounds=args.max_rounds,
        repair_timeout=args.repair_timeout,
        repair_num_predict=args.repair_num_predict,
        repair_num_ctx=args.repair_num_ctx,
        repair_temperature=args.repair_temperature,
        only_apis=args.only_apis,
        force=args.force,
        external_errors_csv=args.external_errors_csv,
        external_errors_jsonl=args.external_errors_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
