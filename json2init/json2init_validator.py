
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[0]
for candidate in [Path.cwd(), ROOT, ROOT.parent]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

from common.model_config import check_model_backend, load_model_config
from common.pipeline_contract import API_DOC_TEXT, API_FULL_NAME, SIGNATURE, read_api_records

try:
    from json2init.deepfuzz_common import build_runtime_object_from_spec, run_smoke_test
except Exception:
    from deepfuzz_common import build_runtime_object_from_spec, run_smoke_test  # type: ignore

SECTION_ARGS = {"args", "arguments", "parameters", "parameter", "inputs", "input"}
SECTION_RETURNS = {"returns", "return", "output", "outputs", "result", "results"}
STOP_SECTIONS = {"raises", "examples", "example", "note", "notes", "warning", "warnings", "see also", "references"}
BAD_PARAM_NAMES = {
    "optional", "required", "parameter", "parameters", "argument", "arguments", "arg", "args",
    "default", "defaults", "note", "notes", "example", "examples", "return", "returns",
    "result", "results", "dtype", "tensor", "tensors", "list", "tuple", "type", "if", "then", "else",
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
    (r"\b(?:tensor|tensors|ndarray|array|bytetensor)\b", "tensor"),
    (r"\bdevice\b", "device"),
    (r"\bdtype\b", "dtype"),
    (r"\bany\b", "any"),
]
DTYPE_NAMES = (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16", "bool", "complex64", "complex128",
)
FIXABLE_CLASSIFICATIONS = {"spec_or_seed_error", "materialization_error", "import_error", "seed_generation_error"}
NON_FIXABLE_CLASSIFICATIONS = {"environment_unsupported", "missing_dependency"}
REPAIRABLE_STAGE3_STATUSES = {"retry", "api_runtime_error"}
REPAIR_PROMPT = """You repair exactly one API JSON spec for Stage 3 base-seed materialization.

Return ONLY valid JSON.
No markdown.
No prose.
No comments.

API Full Name: {api_full_name}

Observed Signature Line:
{sig_line}

Deterministically Supported Parameter Names:
{supported_params}

Current Stage 3 Failure:
{failure_summary}

API Documentation Text:
{api_doc_text}

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
2. Only include parameters from the supported list above.
3. Remove fake parameters from wrapped prose.
4. If a real required parameter appears in docs/signature, include it.
5. If uncertain, leave fields empty instead of guessing.
6. flag must be exactly one of: Required, Optional, "".
7. Put single-parameter rules under that parameter.
8. Put only cross-parameter rules at top-level.
9. Do not invent executable defaults.
10. Keep descriptions short and literal.
11. Favor the simplest seed-friendly constructible type when docs allow multiple compatible choices.
12. Output exactly one JSON object.
"""


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
    return "" if s.lower() in {"null"} else s


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


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
    return bool(d and (
        "(optional)" in d or d.startswith("optional") or " optional" in d or
        " defaults to " in d or d.startswith("defaults to ") or
        " default is " in d or " default: " in d or re.search(r"\bdefault\b", d) or
        "if none" in d
    ))


def extract_size_text(text: str) -> str:
    s = normalize_ws(text)
    if not s:
        return ""
    for pat in [
        r"\b\d+\s*[- ]?d\b", r"\b\d+\s*[- ]?d(?:imension|im)?s?\b", r"\bshape\s*=\s*\([^)]*\)",
        r"\bshape\s+\([^)]*\)", r"\btuple of length \d+\b", r"\blist of length \d+\b", r"\bscalar\b"
    ]:
        m = re.search(pat, s, flags=re.I)
        if m:
            return normalize_ws(m.group(0)).replace("-d", "-D")
    return ""


def normalize_type(type_str: str, desc_text: str = "") -> str:
    text = " ".join(x for x in [clean_text(type_str), clean_text(desc_text)] if x)
    if not text:
        return ""
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


def parse_doc_params(doc: str, allowed_names: Optional[Sequence[str]] = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, bool], Dict[str, str]]:
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
        lines = split_lines(doc)
        first = normalize_ws(lines[0]) if lines else ""
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


def normalize_schema(spec: Dict[str, Any], api_full_name: str, doc: str, signature: str = "") -> Dict[str, Any]:
    module_path, _, api_name = api_full_name.rpartition(".")
    sig_line = clean_text(signature) or find_signature_line(doc)
    sig_params, sig_defaults, _ = parse_signature_params(sig_line)
    doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map = parse_doc_params(doc, allowed_names=sig_params if sig_params else None)
    ret_type_doc, ret_numbers_doc, ret_desc_doc = parse_return_info(doc)
    llm_params = spec.get("params", {}) if isinstance(spec.get("params"), dict) else {}
    supported_names: List[str] = []
    seen = set()
    for p in sig_params + list(doc_type_map.keys()) + list(doc_desc_map.keys()):
        if is_suspicious_param_name(p):
            continue
        if p and p not in seen:
            seen.add(p)
            supported_names.append(p)
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
        "constraints": normalize_string_list(spec.get("constraints")),
    }
    for name in supported_names:
        p = llm_params.get(name, {}) if isinstance(llm_params.get(name), dict) else {}
        desc = clean_text(doc_desc_map.get(name, "")) or clean_text(p.get("description", ""))
        raw_type = clean_text(doc_type_map.get(name, "")) or clean_text(p.get("type", ""))
        if name in sig_defaults:
            default = clean_text(sig_defaults[name])
        elif name in doc_default_map:
            default = clean_text(doc_default_map[name])
        else:
            default = clean_text(p.get("default", ""))
        if name in sig_defaults or doc_optional_map.get(name, False) or clean_text(p.get("flag", "")) == "Optional":
            flag = "Optional"
        else:
            flag = "Required"
        constraints = normalize_string_list(p.get("constraints"))
        out["params"][name] = {
            "type": normalize_type(raw_type, desc),
            "size": clean_text(doc_size_map.get(name, "")) or extract_size_text(clean_text(p.get("size", ""))),
            "default": default,
            "flag": flag,
            "description": desc,
            "dtype_candidates": normalize_string_list(p.get("dtype_candidates")) or extract_dtype_candidates(raw_type, desc, *constraints),
            "enum_values": normalize_string_list(p.get("enum_values")) or extract_enum_values(raw_type, desc, *constraints, param_names=supported_names),
            "constraints": constraints,
        }
    return out


def call_ollama_json(prompt: str, model: str, host: str, timeout: int = 300, temperature: float = 0.0, num_predict: int = 600, num_ctx: int = 4096) -> Dict[str, Any]:
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


def load_ok_api_names(ok_csv: str) -> Set[str]:
    names: Set[str] = set()
    if not ok_csv or not os.path.exists(ok_csv):
        return names
    with open(ok_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            api = str(row.get("api_full_name", "") or "").strip()
            if api:
                names.add(api)
    return names


def load_stage3_issues(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def load_api_filter(path: str) -> Set[str]:
    names: Set[str] = set()
    if not path or not os.path.exists(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    return names


def make_failure_summary(issue: Dict[str, Any]) -> str:
    reasons = issue.get("reasons", []) or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    smoke = issue.get("smoke_test", {}) or {}
    lines = []
    if reasons:
        lines.append("Readiness reasons: " + " | ".join(str(x) for x in reasons))
    if issue.get("materialization_reasons"):
        lines.append("Materialization reasons: " + " | ".join(str(x) for x in issue.get("materialization_reasons", [])))
    if smoke:
        lines.append("Smoke classification: " + str(smoke.get("classification", "")))
        lines.append("Smoke error: " + str(smoke.get("error", "")))
    if issue.get("reason"):
        lines.append("Issue reason: " + str(issue.get("reason", "")))
    return "\n".join(lines)


def build_prompt(api_full_name: str, doc: str, current_spec: Dict[str, Any], failure_summary: str, signature: str = "") -> str:
    sig_line = clean_text(signature) or find_signature_line(doc)
    sig_params, _, _ = parse_signature_params(sig_line)
    doc_type_map, doc_desc_map, _, _, _ = parse_doc_params(doc, allowed_names=sig_params if sig_params else None)
    supported = []
    seen = set()
    for p in sig_params + list(doc_type_map.keys()) + list(doc_desc_map.keys()):
        if is_suspicious_param_name(p):
            continue
        if p not in seen:
            seen.add(p)
            supported.append(p)
    return REPAIR_PROMPT.format(
        api_full_name=api_full_name,
        sig_line=sig_line or "",
        supported_params=", ".join(supported) if supported else "(none reliably parsed)",
        failure_summary=failure_summary,
        api_doc_text=doc,
        current_json=json.dumps(current_spec, ensure_ascii=False, indent=2),
    )


def classify_issue_for_stage3(runtime_obj: Any, smoke: Dict[str, Any]) -> str:
    if not getattr(runtime_obj, "spec_ready", False):
        return "retry"
    if smoke.get("success"):
        return "ready"
    classification = str(smoke.get("classification", "") or "")
    if classification in FIXABLE_CLASSIFICATIONS:
        return "retry"
    if classification == "environment_unsupported":
        return "env_only"
    if classification == "missing_dependency":
        return "missing_dependency"
    if classification == "api_runtime_error":
        return "api_runtime_error"
    return classification or "retry"


def _smoke_test_worker(q: Any, api_full_name: str, spec: Dict[str, Any]) -> None:
    try:
        runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
        q.put(run_smoke_test(runtime_obj))
    except BaseException as e:
        q.put({
            "success": False,
            "classification": "api_runtime_error",
            "error": f"smoke worker exception: {type(e).__name__}: {e}",
        })


def safe_run_smoke_test(api_full_name: str, spec: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_smoke_test_worker, args=(q, api_full_name, spec))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.kill()
        p.join()
        return {
            "success": False,
            "classification": "api_runtime_error",
            "error": f"smoke test timed out after {timeout_sec}s",
        }

    if p.exitcode != 0:
        return {
            "success": False,
            "classification": "seed_generation_error",
            "error": f"smoke test subprocess exited abnormally (exitcode={p.exitcode})",
            "exit_code": p.exitcode,
        }

    try:
        out = q.get_nowait()
        return out if isinstance(out, dict) else {
            "success": False,
            "classification": "api_runtime_error",
            "error": "smoke test subprocess returned invalid result",
        }
    except Exception:
        return {
            "success": False,
            "classification": "api_runtime_error",
            "error": "smoke test subprocess returned no result",
        }


def evaluate_spec(api_full_name: str, spec: Dict[str, Any], smoke_timeout_sec: int) -> Tuple[str, Dict[str, Any], List[str], str]:
    runtime_obj = build_runtime_object_from_spec(api_full_name, spec)
    reasons = list(getattr(runtime_obj, "readiness_reasons", []) or [])
    smoke: Dict[str, Any] = {}
    classification = ""
    if getattr(runtime_obj, "spec_ready", False):
        smoke = safe_run_smoke_test(api_full_name, spec, timeout_sec=smoke_timeout_sec)
        classification = str(smoke.get("classification", "") or "")
        if not smoke.get("success") and classification in FIXABLE_CLASSIFICATIONS:
            msg = f"smoke test failed: {smoke.get('error') or classification}"
            if msg not in reasons:
                reasons.append(msg)
    stage_status = classify_issue_for_stage3(runtime_obj, smoke)
    return stage_status, smoke, reasons, classification


def choose_repair_model(round_idx: int, primary_model: str, fallback_model: str, fallback_after_round: int) -> str:
    if fallback_model and round_idx >= fallback_after_round:
        return fallback_model
    return primary_model or fallback_model


def process(
    spec_dir: str,
    docs_path: str,
    stage3_issues: str,
    outdir: str,
    ok_csv: str = "",
    primary_repair_model: str = "mistral:7b",
    fallback_repair_model: str = "mixtral:8x7b",
    fallback_after_round: int = 3,
    repair_host: str = "http://localhost:11434",
    max_rounds: int = 3,
    repair_timeout: int = 300,
    repair_num_predict: int = 600,
    repair_num_ctx: int = 4096,
    repair_temperature: float = 0.0,
    write_back: bool = True,
    only_stage3_retry: bool = True,
    only_api_list: str = "",
    smoke_timeout_sec: int = 30,
) -> None:
    docs = read_api_docs(docs_path)
    allowed = load_ok_api_names(ok_csv) if ok_csv else None
    selected_api_names = load_api_filter(only_api_list)
    issues = load_stage3_issues(stage3_issues)
    os.makedirs(outdir, exist_ok=True)

    report_rows: List[Dict[str, Any]] = []
    repaired_csv_rows: List[Dict[str, Any]] = []
    retry_csv_rows: List[Dict[str, Any]] = []
    errors_csv_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []

    summary = {
        "total_seen": 0,
        "selected": 0,
        "skipped_not_repairable": 0,
        "env_only": 0,
        "missing_dependency": 0,
        "repaired": 0,
        "repaired_by_normalization": 0,
        "repaired_by_llm": 0,
        "still_retry": 0,
        "still_api_runtime_error": 0,
        "still_env_only": 0,
        "still_missing_dependency": 0,
        "environment_model_unavailable": 0,
        "no_doc": 0,
    }
    model_health_checked = False
    model_health_ok = True
    model_health_message = ""

    for issue in issues:
        summary["total_seen"] += 1
        api_full_name = str(issue.get("api_full_name", "") or "").strip()
        if not api_full_name:
            continue
        if allowed and api_full_name not in allowed:
            continue
        if selected_api_names and api_full_name not in selected_api_names:
            continue

        stage3_status = str(issue.get("stage3_status", "") or "")
        if only_stage3_retry and stage3_status and stage3_status not in REPAIRABLE_STAGE3_STATUSES:
            summary["skipped_not_repairable"] += 1
            continue

        summary["selected"] += 1
        spec_path = str(issue.get("source_spec_json_path", "") or os.path.join(spec_dir, f"{api_full_name}.json"))
        current_spec = load_json(spec_path)
        doc_row = docs.get(api_full_name, {})
        doc = str(doc_row.get(API_DOC_TEXT, "") or "")
        signature = str(doc_row.get(SIGNATURE, "") or "")
        smoke = issue.get("smoke_test", {}) or {}
        smoke_classification = str(smoke.get("classification", "") or issue.get("smoke_classification", "") or "")

        record: Dict[str, Any] = {
            "api_full_name": api_full_name,
            "input_stage3_status": stage3_status,
            "initial_smoke_classification": smoke_classification,
            "source_spec_json_path": spec_path,
            "rounds": 0,
            "repair_model_used": "",
            "status": "",
            "reason": "",
            "repaired_spec_json_path": "",
            "write_back": write_back,
        }

        if not doc:
            summary["no_doc"] += 1
            record.update({"status": "no_doc", "reason": "missing source doc"})
            result_rows.append(record)
            errors_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "no_doc",
                "rounds": 0,
                "repair_model_used": "",
                "reason": "missing source doc",
                "source_spec_json_path": spec_path,
            })
            report_rows.append(record.copy())
            continue

        if smoke_classification in NON_FIXABLE_CLASSIFICATIONS:
            key = "env_only" if smoke_classification == "environment_unsupported" else "missing_dependency"
            summary[key] += 1
            record.update({"status": key, "reason": str(smoke.get("error", "") or smoke_classification)})
            result_rows.append(record)
            report_rows.append(record.copy())
            continue

        current = normalize_schema(current_spec, api_full_name, doc, signature)
        status_eval, smoke_eval, reasons, classification_eval = evaluate_spec(api_full_name, current, smoke_timeout_sec=smoke_timeout_sec)
        record["rounds"] = 1

        if status_eval == "ready":
            if write_back:
                write_json(spec_path, current)
            summary["repaired"] += 1
            summary["repaired_by_normalization"] += 1
            record.update({
                "status": "ready",
                "reason": "deterministic normalization fixed issue",
                "repaired_spec_json_path": spec_path,
            })
            repaired_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "ready",
                "rounds": 1,
                "repair_model_used": "",
                "repaired_spec_json_path": spec_path,
                "source_spec_json_path": spec_path,
            })
            result_rows.append(record)
            report_rows.append({**record, "final_smoke_classification": classification_eval})
            continue

        if status_eval in {"env_only", "missing_dependency"}:
            key = "still_env_only" if status_eval == "env_only" else "still_missing_dependency"
            summary[key] += 1
            record.update({"status": status_eval, "reason": str(smoke_eval.get("error", "") or classification_eval)})
            result_rows.append(record)
            report_rows.append({**record, "final_smoke_classification": classification_eval})
            continue

        current_reasons = reasons or [make_failure_summary(issue)]
        final_status = status_eval
        final_classification = classification_eval
        final_smoke = smoke_eval

        can_try_llm = bool(primary_repair_model or fallback_repair_model)
        if can_try_llm:
            if not model_health_checked:
                model_name = primary_repair_model or fallback_repair_model
                cfg = load_model_config()
                model_health_ok, model_health_message = check_model_backend(model_name, repair_host, backend=cfg.backend)
                model_health_checked = True
            if not model_health_ok:
                summary["environment_model_unavailable"] += 1
                record.update({
                    "status": "environment_model_unavailable",
                    "reason": model_health_message,
                    "repair_model_used": "unavailable",
                })
                errors_csv_rows.append({
                    "api_full_name": api_full_name,
                    "status": "environment_model_unavailable",
                    "rounds": record["rounds"],
                    "repair_model_used": "unavailable",
                    "reason": model_health_message,
                    "source_spec_json_path": spec_path,
                })
                result_rows.append(record)
                report_rows.append({**record, "final_smoke_classification": final_classification})
                continue
            for round_idx in range(2, max_rounds + 1):
                model_name = choose_repair_model(round_idx, primary_repair_model, fallback_repair_model, fallback_after_round)
                if not model_name:
                    break
                record["rounds"] = round_idx
                record["repair_model_used"] = model_name
                failure_summary = " | ".join(current_reasons) if current_reasons else make_failure_summary(issue)
                prompt = build_prompt(api_full_name, doc, current, failure_summary, signature)
                candidate = call_ollama_json(
                    prompt,
                    model_name,
                    repair_host,
                    timeout=repair_timeout,
                    temperature=repair_temperature,
                    num_predict=repair_num_predict,
                    num_ctx=repair_num_ctx,
                )
                current = normalize_schema(candidate, api_full_name, doc, signature)
                final_status, final_smoke, current_reasons, final_classification = evaluate_spec(api_full_name, current, smoke_timeout_sec=smoke_timeout_sec)
                if final_status == "ready":
                    if write_back:
                        write_json(spec_path, current)
                    summary["repaired"] += 1
                    summary["repaired_by_llm"] += 1
                    record.update({
                        "status": "repaired",
                        "reason": "llm-assisted repair succeeded",
                        "repaired_spec_json_path": spec_path,
                    })
                    repaired_csv_rows.append({
                        "api_full_name": api_full_name,
                        "status": "repaired",
                        "rounds": round_idx,
                        "repair_model_used": model_name,
                        "repaired_spec_json_path": spec_path,
                        "source_spec_json_path": spec_path,
                    })
                    break
                if final_status in {"env_only", "missing_dependency"}:
                    break

        if record.get("status") in {"ready", "repaired"}:
            result_rows.append(record)
            report_rows.append({**record, "final_smoke_classification": final_classification})
            continue

        if final_status == "env_only":
            summary["still_env_only"] += 1
            record.update({"status": "env_only", "reason": str(final_smoke.get("error", "") or final_classification)})
        elif final_status == "missing_dependency":
            summary["still_missing_dependency"] += 1
            record.update({"status": "missing_dependency", "reason": str(final_smoke.get("error", "") or final_classification)})
        elif final_status == "api_runtime_error":
            summary["still_api_runtime_error"] += 1
            reason_text = str(final_smoke.get("error", "") or final_classification)
            record.update({"status": "api_runtime_error", "reason": reason_text})
            retry_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "api_runtime_error",
                "rounds": record["rounds"],
                "repair_model_used": record["repair_model_used"],
                "reason": reason_text,
                "source_spec_json_path": spec_path,
            })
            errors_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "api_runtime_error",
                "rounds": record["rounds"],
                "repair_model_used": record["repair_model_used"],
                "reason": reason_text,
                "source_spec_json_path": spec_path,
            })
        else:
            summary["still_retry"] += 1
            reason_text = " | ".join(current_reasons) if current_reasons else make_failure_summary(issue)
            record.update({"status": "retry", "reason": reason_text})
            retry_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "retry",
                "rounds": record["rounds"],
                "repair_model_used": record["repair_model_used"],
                "reason": reason_text,
                "source_spec_json_path": spec_path,
            })
            errors_csv_rows.append({
                "api_full_name": api_full_name,
                "status": "retry",
                "rounds": record["rounds"],
                "repair_model_used": record["repair_model_used"],
                "reason": reason_text,
                "source_spec_json_path": spec_path,
            })

        result_rows.append(record)
        report_rows.append({**record, "final_smoke_classification": final_classification})

    write_csv(
        os.path.join(outdir, "repaired.csv"),
        repaired_csv_rows,
        ["api_full_name", "status", "rounds", "repair_model_used", "repaired_spec_json_path", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "retry_only.csv"),
        retry_csv_rows,
        ["api_full_name", "status", "rounds", "repair_model_used", "reason", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "errors.csv"),
        errors_csv_rows,
        ["api_full_name", "status", "rounds", "repair_model_used", "reason", "source_spec_json_path"],
    )
    write_csv(
        os.path.join(outdir, "validation_report.csv"),
        report_rows,
        [
            "api_full_name",
            "input_stage3_status",
            "initial_smoke_classification",
            "status",
            "rounds",
            "repair_model_used",
            "reason",
            "repaired_spec_json_path",
            "source_spec_json_path",
            "final_smoke_classification",
            "write_back",
        ],
    )
    with open(os.path.join(outdir, "repair_results.jsonl"), "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retry_api_names = sorted({str(r["api_full_name"]) for r in retry_csv_rows})
    retry_list_path = os.path.join(outdir, "retry_api_list.txt")
    with open(retry_list_path, "w", encoding="utf-8") as f:
        for api_name in retry_api_names:
            f.write(api_name + "\n")

    write_json(os.path.join(outdir, "summary.json"), summary)
    print(
        f"[json2init-validator] selected={summary['selected']} repaired={summary['repaired']} "
        f"still_retry={summary['still_retry']} still_api_runtime_error={summary['still_api_runtime_error']} "
        f"env_only={summary['env_only']} missing_dependency={summary['missing_dependency']}"
    )
    print(f"[json2init-validator] repaired: {os.path.join(outdir, 'repaired.csv')}")
    print(f"[json2init-validator] retry_only: {os.path.join(outdir, 'retry_only.csv')}")
    print(f"[json2init-validator] errors: {os.path.join(outdir, 'errors.csv')}")
    print(f"[json2init-validator] report: {os.path.join(outdir, 'validation_report.csv')}")
    print(f"[json2init-validator] summary: {os.path.join(outdir, 'summary.json')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 json2init validator/repairer for fixable spec/seed issues")
    ap.add_argument("--allow-deprecated-output-contract", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--api-csv", required=True, help="CSV/XLSX with api_full_name, api_doc_text")
    ap.add_argument("--stage3-issues", required=True, help="Stage 3 issues.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ok-csv", default="", help="Optional Stage 2 ok.csv allowlist")
    ap.add_argument("--primary-repair-model", default="mistral:7b")
    ap.add_argument("--repair-model", dest="primary_repair_model", default=argparse.SUPPRESS, help="Backward-compatible alias for --primary-repair-model")
    ap.add_argument("--fallback-repair-model", default="mixtral:8x7b")
    ap.add_argument("--fallback-after-round", type=int, default=3)
    ap.add_argument("--repair-host", default="http://localhost:11434")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--repair-timeout", type=int, default=300)
    ap.add_argument("--repair-num-predict", type=int, default=600)
    ap.add_argument("--repair-num-ctx", type=int, default=4096)
    ap.add_argument("--repair-temperature", type=float, default=0.0)
    ap.add_argument("--smoke-timeout-sec", type=int, default=30)
    ap.add_argument("--no-write-back", action="store_true")
    ap.add_argument("--only-stage3-retry", action="store_true", help="Repair only rows whose stage3_status is retry or api_runtime_error")
    ap.add_argument("--only-api-list", default="", help="Optional newline-delimited list of api_full_name values to process")
    args = ap.parse_args()
    if not args.allow_deprecated_output_contract:
        print(
            "[json2init-validator] deprecated: use scripts/repair.py so Stage 3 failures stay in errors.csv only.",
            file=sys.stderr,
        )
        return 2
    process(
        spec_dir=args.spec_dir,
        docs_path=args.api_csv,
        stage3_issues=args.stage3_issues,
        outdir=args.outdir,
        ok_csv=args.ok_csv or "",
        primary_repair_model=args.primary_repair_model,
        fallback_repair_model=args.fallback_repair_model,
        fallback_after_round=args.fallback_after_round,
        repair_host=args.repair_host,
        max_rounds=args.max_rounds,
        repair_timeout=args.repair_timeout,
        repair_num_predict=args.repair_num_predict,
        repair_num_ctx=args.repair_num_ctx,
        repair_temperature=args.repair_temperature,
        write_back=not args.no_write_back,
        only_stage3_retry=args.only_stage3_retry,
        only_api_list=args.only_api_list,
        smoke_timeout_sec=args.smoke_timeout_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
