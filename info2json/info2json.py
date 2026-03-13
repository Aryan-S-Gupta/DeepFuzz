#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    import pandas as pd  # optional, only for xlsx
except Exception:
    pd = None


PROMPT_TEMPLATE = """You are an expert API reverse-engineering assistant.
Convert the API documentation into exactly ONE JSON object for downstream input initialization,
mutation-rule setup, and execution-time validation.

Return ONLY valid JSON. No markdown. No prose. No comments.

API Full Name: {api_full_name}

Observed Signature Line (may be empty):
{sig_line}

Parameter Names Supported by Deterministic Parsing:
{supported_params}

API Documentation Text:
\"\"\"
{api_doc_text}
\"\"\"

Required JSON schema:
{{
  "api_name": "",
  "module_path": "",
  "params": {{
    "<param_name>": {{
      "type": "",
      "size": "",
      "default": "",
      "flag": "",
      "description": ""
    }}
  }},
  "output": {{
    "type": "",
    "numbers": "",
    "description": ""
  }},
  "constraints": [""]
}}

Rules:
1. Use only information explicitly stated in the documentation text or the observed signature line.
2. Do not include implicit parameters like self or cls.
3. Do not invent parameter names.
4. Only include parameters that are supported by the observed signature line or the deterministic parsing result above.
5. Never treat words like optional, required, parameter, argument, input, output, return, default, note, example, dtype, tensor, list, tuple, or type as parameter names unless they explicitly appear as real parameter names.
6. Every parameter must be placed under "params".
7. flag must be exactly one of: "Required", "Optional", "".
8. Mark a parameter as "Optional" only when the signature or documentation explicitly indicates optionality, a default value, or wording like "(optional)" or "Defaults to ...".
9. If a field is unknown, use the empty string "".
10. constraints must contain short, mutation-useful validation rules only when the documentation explicitly supports them.
11. Good constraint examples:
   - "axis must satisfy -rank(input) <= axis < rank(input)"
   - "x and y must have broadcast-compatible shapes"
   - "minval must be less than maxval"
   - "input rank must be at least 2"
   - "indices dtype must be int32 or int64"
   - "indices values must be in range [0, params.shape[axis])"
12. Do not invent hidden semantics. If the documentation does not support a rule, omit it.
13. Keep descriptions short and documentation-grounded.
14. Output exactly one JSON object and nothing else.
"""


def read_apis(input_path: str) -> Iterable[Tuple[str, str]]:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".csv":
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                api = str(row.get("api_full_name", "") or "").strip()
                doc = str(row.get("api_doc_text", "") or "")
                if api:
                    yield api, doc
        return

    if suffix in {".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError("pandas/openpyxl is required for Excel input")
        df = pd.read_excel(input_path).fillna("")
        for _, row in df.iterrows():
            api = str(row.get("api_full_name", "") or "").strip()
            doc = str(row.get("api_doc_text", "") or "")
            if api:
                yield api, doc
        return

    raise ValueError(f"Unsupported input file: {input_path}")


def sanitize_filename(api_full_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", api_full_name.strip())
    return f"{safe}.json"


def call_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 300,
    temperature: float = 0.0,
    num_predict: int = 700,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return str(data["response"])


def blank_spec() -> Dict[str, Any]:
    return {
        "api_name": "",
        "module_path": "",
        "params": {},
        "output": {"type": "", "numbers": "", "description": ""},
        "constraints": [],
    }


def extract_json(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text

    raw = str(text or "").strip()
    if not raw:
        return blank_spec()

    if raw.startswith("```"):
        parts = raw.split("```", 2)
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

    candidates = [raw]

    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        try:
            inner = json.loads(raw)
            if isinstance(inner, str):
                candidates.append(inner)
            elif isinstance(inner, dict):
                return inner
        except Exception:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return blank_spec()


SECTION_ARGS = {"args", "arguments", "parameters", "parameter", "inputs", "input"}
SECTION_RETURNS = {"returns", "return", "output", "outputs", "result", "results"}
STOP_SECTIONS = {"raises", "examples", "example", "note", "notes", "warning", "warnings", "see also", "references"}
BAD_PARAM_NAMES = {
    "optional", "required", "parameter", "parameters", "argument", "arguments", "arg", "args",
    "input", "inputs", "output", "outputs", "return", "returns", "result", "results",
    "default", "defaults", "note", "notes", "example", "examples",
}
BAD_RETURN_TYPES = {"note", "notes", "example", "examples", "warning", "warnings"}


SPECIFIC_TYPE_PATTERNS = [
    re.compile(r"\b(?:raggedtensor|sparsetensor|tensorarray|dataset|variable|indexedslices)\b", re.I),
    re.compile(r"\b(?:list|tuple|sequence)\s+of\b", re.I),
    re.compile(r"\b[123]-d\b", re.I),
    re.compile(r"\btf\.[A-Za-z_][\w.]*\b"),
    re.compile(r"\btorch\.[A-Za-z_][\w.]*\b"),
]

TYPE_CANON = [
    (r"\b(sparsetensor|raggedtensor|tensor|tensors|ndarray|array)\b", "tensor"),
    (r"\b(float|double|float16|float32|float64|bfloat16|half|real)\b", "float"),
    (r"\b(int|integer|int8|int16|int32|int64|uint8|uint16|uint32|uint64|long|short)\b", "int"),
    (r"\b(bool|boolean)\b", "bool"),
    (r"\b(str|string|bytes)\b", "string"),
    (r"\b(list|lists)\b", "list"),
    (r"\b(tuple|tuples)\b", "tuple"),
    (r"\b(sequence|iterable)\b", "sequence"),
    (r"\b(dict|dictionary|mapping)\b", "dict"),
    (r"\b(dtype)\b", "dtype"),
    (r"\b(device|cpu|cuda|gpu|tpu)\b", "device"),
    (r"\b(callable|function|fn)\b", "callable"),
    (r"\b(number|numeric|scalar)\b", "number"),
]

INT_LIKE_NAMES = {
    "dim", "dims", "axis", "axes", "ndim", "rank", "k", "n", "m", "num", "num_classes",
    "num_layers", "steps", "bins", "size", "length", "depth", "width", "height", "count", "index", "indices",
}
BOOL_LIKE_NAMES = {
    "keepdim", "keepdims", "training", "inplace", "transpose", "transpose_a", "transpose_b",
    "normalize", "normalized", "sorted", "exclusive", "reverse", "adjoint", "adjoint_a", "adjoint_b",
    "conjugate", "lower", "upper", "center", "enabled", "a_is_sparse", "b_is_sparse", "grad_a", "grad_b",
}
FLOAT_LIKE_NAMES = {
    "eps", "epsilon", "alpha", "beta", "gamma", "sigma", "stddev", "std", "mean", "momentum",
    "learning_rate", "lr", "prob", "probability", "p",
}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def split_lines(doc: str) -> List[str]:
    return str(doc or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def is_suspicious_param_name(name: str) -> bool:
    n = normalize_ws(name).lower().strip("*` ")
    if not n:
        return True
    if n in {"self", "cls"}:
        return True
    if n in BAD_PARAM_NAMES:
        return True
    if len(n) > 64:
        return True
    return False


def find_signature_line(doc: str) -> str:
    lines = [ln.strip() for ln in split_lines(doc) if ln.strip()]
    for line in lines[:20]:
        if "(" in line and ")" in line and not line.endswith(":"):
            return line
    for line in lines:
        if "(" in line and ")" in line and not line.endswith(":"):
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

        ret = None
        if "->" in sig_line:
            ret = normalize_ws(sig_line.split("->", 1)[1])
        return params, defaults, ret
    except Exception:
        return [], {}, None


def extract_size_text(text: str) -> str:
    s = normalize_ws(text)
    if not s:
        return ""

    m = re.search(r"\bshape\s*=\s*\([^)]*\)", s, flags=re.I)
    if m:
        return normalize_ws(m.group(0))

    m = re.search(r"\bshape\s+\([^)]*\)", s, flags=re.I)
    if m:
        return normalize_ws(m.group(0))

    m = re.search(r"\brank\s*[<>]=?\s*\d+", s, flags=re.I)
    if m:
        return normalize_ws(m.group(0))

    m = re.search(r"\b\d+\s*[- ]?d(?:imension|im)?s?\b", s, flags=re.I)
    if m:
        return normalize_ws(m.group(0))

    m = re.search(r"\(([^)]*(?:shape|rank|ndim|dimension|dimensions)[^)]*)\)", s, flags=re.I)
    if m:
        return normalize_ws(f"({m.group(1)})")

    return ""

def parse_doc_params(doc: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, bool]]:
    """Return name->type, name->desc, name->size, name->is_optional."""
    lines = split_lines(doc)
    type_map: Dict[str, str] = {}
    desc_map: Dict[str, str] = {}
    size_map: Dict[str, str] = {}
    optional_map: Dict[str, bool] = {}

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
            name = m.group(1).lstrip("*")
            if is_suspicious_param_name(name):
                continue
            desc = normalize_ws(m.group(2))
            desc_map[name] = desc
            optional_map[name] = explicit_optional_from_text(desc)
            size = extract_size_text(desc)
            if size:
                size_map[name] = size
            continue

        m = p_rst_type.match(line)
        if m:
            name = m.group(1).lstrip("*")
            if is_suspicious_param_name(name):
                continue
            typ = normalize_ws(m.group(2))
            type_map[name] = typ
            optional_map[name] = optional_map.get(name, False) or explicit_optional_from_text(typ)
            continue

        if not in_args:
            continue

        m = p_google.match(line)
        if m:
            name = m.group(1).lstrip("*")
            if is_suspicious_param_name(name):
                current_name = None
                continue
            meta = normalize_ws(m.group(3))
            desc = normalize_ws(m.group(4))
            current_name = name
            if meta:
                type_map[name] = meta
                size = extract_size_text(meta)
                if size:
                    size_map[name] = size
            if desc:
                desc_map[name] = desc
                size = extract_size_text(desc)
                if size:
                    size_map[name] = size
            optional_map[name] = explicit_optional_from_text(meta) or explicit_optional_from_text(desc)
            continue

        m = p_numpy.match(line)
        if m and idx + 1 < len(lines):
            name = m.group(1).lstrip("*")
            if is_suspicious_param_name(name):
                current_name = None
                continue
            meta = normalize_ws(m.group(2))
            current_name = name
            if meta:
                type_map[name] = meta
                size = extract_size_text(meta)
                if size:
                    size_map[name] = size
            optional_map[name] = explicit_optional_from_text(meta)
            continue

        if current_name and (line.startswith(" ") or line.startswith("\t")):
            extra = normalize_ws(line)
            if extra:
                prev = desc_map.get(current_name, "")
                desc_map[current_name] = normalize_ws((prev + " " + extra).strip())
                optional_map[current_name] = optional_map.get(current_name, False) or explicit_optional_from_text(extra)
                size = extract_size_text(extra)
                if size and current_name not in size_map:
                    size_map[current_name] = size

    return type_map, desc_map, size_map, optional_map


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
                candidate_type = normalize_ws(m.group(1))
                if candidate_type.lower() not in BAD_RETURN_TYPES:
                    ret_type = candidate_type
                    ret_desc_parts.append(normalize_ws(m.group(2)))
                    continue

            if len(stripped.split()) <= 8 and not ret_desc_parts and not re.match(r"^(A|An|The)\b", stripped):
                ret_type = normalize_ws(stripped)
            else:
                ret_desc_parts.append(normalize_ws(stripped))
        else:
            ret_desc_parts.append(normalize_ws(stripped))

    if not ret_desc_parts:
        m = re.search(r":returns?\s*:\s*(.+)", doc)
        if m:
            ret_desc_parts.append(normalize_ws(m.group(1)))

    if not ret_type:
        sig = find_signature_line(doc)
        if "->" in sig:
            ret_type = normalize_ws(sig.split("->", 1)[1])

    ret_desc = normalize_ws(" ".join(ret_desc_parts))
    if not ret_type:
        ret_type = infer_return_type_from_desc(ret_desc)
    ret_numbers = infer_output_numbers(ret_desc)
    return ret_type, ret_numbers, ret_desc


def clean_text(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    s = normalize_ws(v)
    return "" if s.lower() in {"none", "null"} else s


def explicit_optional_from_text(text: str) -> bool:
    d = clean_text(text).lower()
    if not d:
        return False
    return bool(
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


def should_preserve_specific_type(type_str: str) -> bool:
    s = clean_text(type_str)
    if not s:
        return False
    for rx in SPECIFIC_TYPE_PATTERNS:
        if rx.search(s):
            return True
    if re.search(r"\b[A-Z][A-Za-z0-9_]+\b", s) and s.lower() not in {
        "tensor", "string", "bool", "int", "float", "list", "tuple", "sequence", "dict", "dtype", "device", "number"
    }:
        return True
    return False


def normalize_type(type_str: str, name: str = "", default_val: Any = None, desc: str = "") -> str:
    s = clean_text(type_str)
    if s and should_preserve_specific_type(s):
        return s
    if s:
        text = s.lower()
        found: List[str] = []
        for pat, label in TYPE_CANON:
            if re.search(pat, text) and label not in found:
                found.append(label)
        if found:
            return "|".join(found)
        return s

    extra = clean_text(desc).lower()
    if extra and should_preserve_specific_type(extra):
        return clean_text(desc)
    if extra:
        found = []
        for pat, label in TYPE_CANON:
            if re.search(pat, extra) and label not in found:
                found.append(label)
        if found:
            return "|".join(found)

    n = (name or "").strip().lower()
    if n == "dtype":
        return "dtype"
    if n == "device":
        return "device"
    if n in BOOL_LIKE_NAMES:
        return "bool"
    if n in INT_LIKE_NAMES:
        return "int"
    if n in FLOAT_LIKE_NAMES:
        return "float"

    if isinstance(default_val, bool):
        return "bool"
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        return "int"
    if isinstance(default_val, float):
        return "float"
    if isinstance(default_val, str) and default_val:
        if default_val in {"True", "False"}:
            return "bool"
        if re.fullmatch(r"[-+]?\d+", default_val):
            return "int"
        if re.fullmatch(r"[-+]?(?:\d*\.\d+|\d+\.\d*)(?:[eE][-+]?\d+)?", default_val):
            return "float"
        return "string"
    return ""


def infer_return_type_from_desc(desc: str) -> str:
    d = clean_text(desc)
    if not d:
        return ""
    return normalize_type(d)


def infer_output_numbers(desc: str) -> str:
    d = (desc or "").lower()
    if not d:
        return ""
    if re.search(r"\btuple of two\b|\btwo values\b|\b2 values\b|\breturns 2\b", d):
        return "2"
    if re.search(r"\btuple of three\b|\bthree values\b|\b3 values\b|\breturns 3\b", d):
        return "3"
    if re.search(r"\breturns? a tuple\b|\breturns? a list\b", d):
        return "multiple"
    return "1"


def normalize_llm_params(raw: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("param") or item.get("arg") or item.get("parameter")
            if name:
                out[str(name)] = item
        return out
    return {}


def infer_optional_from_doc(type_text: str, desc: str, default_text: str = "") -> bool:
    text = " ".join([clean_text(type_text), clean_text(desc), clean_text(default_text)])
    return explicit_optional_from_text(text)


PARAM_RELATION_PATTERNS = [
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+positive\b", re.I), "{0} must be positive"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+non[- ]negative\b", re.I), "{0} must be non-negative"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+negative\b", re.I), "{0} must be negative"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+nonzero\b", re.I), "{0} must be nonzero"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+sorted\b", re.I), "{0} must be sorted"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+numeric\b", re.I), "{0} must be numeric"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\s+must\s+have\s+the\s+same\s+shape\b", re.I), "{0} and {1} must have the same shape"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\s+must\s+have\s+broadcast[- ]compatible\s+shapes\b", re.I), "{0} and {1} must have broadcast-compatible shapes"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+have\s+the\s+same\s+dtype\s+as\s+([A-Za-z_]\w*)\b", re.I), "{0} must have the same dtype as {1}"),
    (re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+less\s+than\s+([A-Za-z_]\w*)\b", re.I), "{0} must be less than {1}"),
    (re.compile(r"\b([A-Za-z_]\w*)\s*<\s*([A-Za-z_]\w*)\b"), "{0} must be less than {1}"),
    (re.compile(r"\b([A-Za-z_]\w*)\s*<=\s*([A-Za-z_]\w*)\b"), "{0} must be less than or equal to {1}"),
    (re.compile(r"\b([A-Za-z_]\w*)\s*>\s*([A-Za-z_]\w*)\b"), "{0} must be greater than {1}"),
    (re.compile(r"\b([A-Za-z_]\w*)\s*>=\s*([A-Za-z_]\w*)\b"), "{0} must be greater than or equal to {1}"),
]


def normalize_range_constraint_text(text: str) -> str:
    s = clean_text(text).strip("`")
    if s and s.startswith("[") and not s.endswith(")") and not s.endswith("]"):
        s += ")"
    return s


def extract_constraints_from_doc(doc: str, param_names: Sequence[str]) -> List[str]:
    lines = [normalize_ws(ln) for ln in split_lines(doc) if normalize_ws(ln)]
    text = "\n".join(lines)
    low = text.lower()
    params = set(param_names)
    out: List[str] = []

    def add(rule: str) -> None:
        rule = normalize_ws(rule)
        if rule and rule not in out:
            out.append(rule)

    for rx, tmpl in PARAM_RELATION_PATTERNS:
        for m in rx.finditer(text):
            names = [g for g in m.groups() if isinstance(g, str)]
            if names and not all((n in params or n.lower() in {"input", "x", "y", "value", "indices", "shape", "dtype", "params", "axis"}) for n in names):
                continue
            add(tmpl.format(*names))

    m = re.search(r"at least\s+(\d+)\s*[- ]?d(?:imension|im)?s?", low)
    if m:
        add(f"input rank must be at least {m.group(1)}")

    m = re.search(r"rank\s+at\s+least\s+(\d+)", low)
    if m:
        add(f"input rank must be at least {m.group(1)}")

    if re.search(r"broadcast(?:ing)?", low):
        mentioned = [p for p in param_names if re.search(rf"\b{re.escape(p)}\b", low)]
        if len(mentioned) >= 2:
            add(f"{mentioned[0]} and {mentioned[1]} must have broadcast-compatible shapes")

    if re.search(r"same\s+shape", low):
        mentioned = [p for p in param_names if re.search(rf"\b{re.escape(p)}\b", low)]
        if len(mentioned) >= 2:
            add(f"{mentioned[0]} and {mentioned[1]} must have the same shape")

    if re.search(r"same\s+dtype|matching\s+dtype", low):
        mentioned = [p for p in param_names if re.search(rf"\b{re.escape(p)}\b", low)]
        if len(mentioned) >= 2:
            add(f"{mentioned[0]} and {mentioned[1]} must have compatible dtypes")

    if re.search(r"sorted", low):
        for p in param_names:
            if re.search(rf"\b{re.escape(p)}\b", low):
                add(f"{p} must be sorted")
                break

    if re.search(r"axis|dimension|dim", low):
        axis_name = next((p for p in param_names if p.lower() in {"axis", "dim", "dims", "dimension"}), "")
        input_name = next((p for p in param_names if p.lower() in {"input", "x", "tensor", "values", "params", "flat_values"}), "input")
        if axis_name and re.search(r"\b(in range|within|valid)\b.*\baxis|\baxis\b.*\b(rank|ndim|dimensions)\b|\bdim\b.*\b(rank|ndim|dimensions)\b", low):
            add(f"{axis_name} must satisfy -rank({input_name}) <= {axis_name} < rank({input_name})")

    if re.search(r"non-?negative|greater than or equal to 0|>=\s*0", low):
        for p in param_names:
            if re.search(rf"\b{re.escape(p)}\b", low):
                add(f"{p} must be non-negative")

    if re.search(r"greater than 0|positive|>\s*0", low):
        for p in param_names:
            if re.search(rf"\b{re.escape(p)}\b", low):
                add(f"{p} must be positive")

    for p in param_names:
        desc_match = re.search(rf"\b{re.escape(p)}\s*:\s*(.+?)(?=(?:\n\S)|\Z)", text, flags=re.I | re.S)
        desc_text = desc_match.group(1) if desc_match else text
        desc_low = desc_text.lower()

        type_choices = re.findall(r"`?(int32|int64|float16|float32|float64|bool|string|complex64|complex128)`?", desc_text, flags=re.I)
        if type_choices and ("one of the following types" in desc_low or "must be one of" in desc_low):
            ordered = []
            for t in type_choices:
                tl = t.lower()
                if tl not in ordered:
                    ordered.append(tl)
            add(f"{p} dtype must be {' or '.join(ordered)}")

        rng = re.search(r"must\s+be\s+in\s+range\s+(`?\[[^\n`]+(?:\)|\])`?)", desc_text, flags=re.I)
        if rng:
            range_text = normalize_range_constraint_text(rng.group(1))
            add(f"{p} values must be in range {range_text}")

        same_as = re.search(r"same\s+type\s+as\s+`?([A-Za-z_][\w\.]*)`?", desc_text, flags=re.I)
        if same_as:
            ref = same_as.group(1)
            if ref != p:
                add(f"output type must match {ref}")

    return out


def build_prompt(api_full_name: str, doc: str) -> str:
    sig_line = find_signature_line(doc)
    sig_params, _, _ = parse_signature_params(sig_line)
    doc_type_map, doc_desc_map, _, _ = parse_doc_params(doc)

    supported = []
    seen = set()
    for p in sig_params + list(doc_type_map.keys()) + list(doc_desc_map.keys()):
        if is_suspicious_param_name(p):
            continue
        if p not in seen:
            supported.append(p)
            seen.add(p)

    supported_text = ", ".join(supported) if supported else "(none reliably parsed)"
    sig_text = sig_line or ""
    return PROMPT_TEMPLATE.format(
        api_full_name=api_full_name,
        sig_line=sig_text,
        supported_params=supported_text,
        api_doc_text=doc,
    )


def normalize_schema(spec: Dict[str, Any], api_full_name: str, doc: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        spec = blank_spec()

    module_path, _, api_name = api_full_name.rpartition(".")
    sig_line = find_signature_line(doc)
    sig_params, sig_defaults, sig_ret = parse_signature_params(sig_line)
    doc_type_map, doc_desc_map, doc_size_map, doc_optional_map = parse_doc_params(doc)
    ret_type_doc, ret_numbers_doc, ret_desc_doc = parse_return_info(doc)

    out = blank_spec()
    out["api_name"] = api_name or api_full_name
    out["module_path"] = module_path

    llm_params = normalize_llm_params(spec.get("params"))

    supported_names: List[str] = []
    seen = set()
    for p in sig_params + list(doc_type_map.keys()) + list(doc_desc_map.keys()):
        if is_suspicious_param_name(p):
            continue
        if p and p not in seen:
            supported_names.append(p)
            seen.add(p)

    if not supported_names:
        for p, info in llm_params.items():
            if is_suspicious_param_name(p):
                continue
            desc = clean_text(info.get("description", ""))
            if desc:
                supported_names.append(p)

    final_params: Dict[str, Dict[str, str]] = {}
    for name in supported_names:
        llm_p = llm_params.get(name, {}) if isinstance(llm_params.get(name, {}), dict) else {}
        default_val = sig_defaults.get(name)
        default_str = "" if default_val is None else str(default_val)

        p_desc = clean_text(doc_desc_map.get(name, "")) or clean_text(llm_p.get("description", ""))
        raw_type_text = clean_text(doc_type_map.get(name, "")) or clean_text(llm_p.get("type", ""))
        p_type = normalize_type(raw_type_text, name=name, default_val=default_val, desc=p_desc)
        p_size = clean_text(doc_size_map.get(name, "")) or extract_size_text(clean_text(llm_p.get("size", "")))

        if name in sig_defaults:
            flag = "Optional"
        elif name in sig_params:
            flag = "Required"
        elif infer_optional_from_doc(raw_type_text, p_desc, default_str) or doc_optional_map.get(name, False):
            flag = "Optional"
        elif default_str:
            flag = "Optional"
        else:
            flag = "Required"

        final_params[name] = {
            "type": p_type,
            "size": p_size,
            "default": default_str,
            "flag": flag,
            "description": p_desc,
        }

    raw_output = spec.get("output") if isinstance(spec.get("output"), dict) else {}
    output_type_raw = ret_type_doc or clean_text(raw_output.get("type", ""))
    out["output"] = {
        "type": normalize_type(output_type_raw, desc=ret_desc_doc) or clean_text(sig_ret or ""),
        "numbers": clean_text(raw_output.get("numbers", "")) or ret_numbers_doc,
        "description": ret_desc_doc or clean_text(raw_output.get("description", "")),
    }

    llm_constraints = spec.get("constraints")
    if isinstance(llm_constraints, str):
        llm_constraints = [llm_constraints]
    elif not isinstance(llm_constraints, list):
        llm_constraints = []

    constraints: List[str] = []
    for c in llm_constraints:
        cc = clean_text(c)
        if cc:
            constraints.append(cc)

    for c in extract_constraints_from_doc(doc, list(final_params.keys())):
        if c not in constraints:
            constraints.append(c)

    out["params"] = final_params
    out["constraints"] = constraints
    return out


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_summary(path: str, summary: Dict[str, Any]) -> None:
    write_json(path, summary)


def process_one(
    api_full_name: str,
    api_doc_text: str,
    outdir: str,
    model: str,
    host: str,
    timeout: int,
    retries: int,
    sleep: float,
) -> Tuple[bool, str]:
    prompt = build_prompt(api_full_name=api_full_name, doc=api_doc_text)
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            raw = call_ollama(prompt, model=model, host=host, timeout=timeout)
            parsed = extract_json(raw)
            normalized = normalize_schema(parsed, api_full_name, api_doc_text)
            out_path = os.path.join(outdir, sanitize_filename(api_full_name))
            write_json(out_path, normalized)
            if sleep > 0:
                time.sleep(sleep)
            return True, out_path
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    return False, last_err


def load_existing_summary(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "failures": [],
    }


def process_file(
    input_path: str,
    outdir: str,
    model: str,
    host: str,
    timeout: int = 300,
    limit: int = 0,
    retries: int = 3,
    sleep: float = 0.0,
    overwrite: bool = False,
    combined_out: Optional[str] = None,
) -> None:
    items = list(read_apis(input_path))
    if limit > 0:
        items = items[:limit]

    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "summary.json")
    failures_csv = os.path.join(outdir, "failures.csv")
    summary = load_existing_summary(summary_path)

    processed_specs: List[Dict[str, Any]] = []
    if combined_out and os.path.exists(combined_out):
        try:
            with open(combined_out, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    processed_specs = loaded
        except Exception:
            processed_specs = []

    failure_rows: List[Dict[str, str]] = []
    if os.path.exists(failures_csv):
        with open(failures_csv, "r", encoding="utf-8", newline="") as f:
            failure_rows = list(csv.DictReader(f))

    for idx, (api, doc) in enumerate(items, start=1):
        out_path = os.path.join(outdir, sanitize_filename(api))
        if os.path.exists(out_path) and not overwrite:
            print(f"[skip] {idx}/{len(items)} {api}")
            continue

        ok, msg = process_one(
            api_full_name=api,
            api_doc_text=doc,
            outdir=outdir,
            model=model,
            host=host,
            timeout=timeout,
            retries=retries,
            sleep=sleep,
        )

        summary["processed"] = int(summary.get("processed", 0)) + 1

        if ok:
            summary["succeeded"] = int(summary.get("succeeded", 0)) + 1
            print(f"[ok] {idx}/{len(items)} {api} -> {msg}")
            if combined_out:
                try:
                    with open(msg, "r", encoding="utf-8") as f:
                        processed_specs.append(json.load(f))
                    write_json(combined_out, processed_specs)
                except Exception:
                    pass
        else:
            summary["failed"] = int(summary.get("failed", 0)) + 1
            failure = {"api_full_name": api, "error": msg}
            summary.setdefault("failures", []).append(failure)
            failure_rows.append(failure)
            print(f"[error] {idx}/{len(items)} {api}: {msg}")
            with open(failures_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["api_full_name", "error"])
                w.writeheader()
                w.writerows(failure_rows)

        summary.update({
            "input": input_path,
            "outdir": outdir,
            "model": model,
            "host": host,
            "total_requested": len(items),
        })
        write_summary(summary_path, summary)

    print(
        f"[done] total={len(items)} processed={summary.get('processed', 0)} "
        f"succeeded={summary.get('succeeded', 0)} failed={summary.get('failed', 0)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert representative API docs into per-API JSON specs using Ollama")
    ap.add_argument("--input", required=True, help="CSV/XLSX with api_full_name, api_doc_text")
    ap.add_argument("--outdir", required=True, help="Output directory for per-API JSON files")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--combined-out", default="", help="Optional combined JSON list path")
    args = ap.parse_args()

    process_file(
        input_path=args.input,
        outdir=args.outdir,
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        limit=args.limit,
        retries=args.retries,
        sleep=args.sleep,
        overwrite=args.overwrite,
        combined_out=args.combined_out or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
