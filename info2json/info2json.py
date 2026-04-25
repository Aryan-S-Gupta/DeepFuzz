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
    import pandas as pd
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
      "description": "",
      "constraints": [""]
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
10. Put single-parameter rules inside that parameter's "constraints".
11. Use top-level "constraints" only for cross-parameter, object-level, or output-level rules.
12. If a rule mentions exactly one real parameter, do not place it only at top level.
13. Good parameter-constraint examples:
   - "axis must satisfy -rank(input) <= axis < rank(input)"
   - "indices dtype must be int32 or int64"
   - "indices values must be in range [0, params.shape[axis])"
   - "input must be 4-D"
14. Good top-level constraint examples:
   - "x and y must have broadcast-compatible shapes"
   - "minval must be less than maxval"
15. Do not invent hidden semantics. If the documentation does not support a rule, omit it.
16. Keep descriptions short and documentation-grounded.
17. Output exactly one JSON object and nothing else.
"""


SECTION_ARGS = {"args", "arguments", "parameters", "parameter", "inputs", "input"}
SECTION_RETURNS = {"returns", "return", "output", "outputs", "result", "results"}
STOP_SECTIONS = {
    "raises", "examples", "example", "note", "notes", "warning", "warnings",
    "see also", "references"
}
BAD_PARAM_NAMES = {
    "optional", "required", "parameter", "parameters", "argument", "arguments", "arg", "args",
    "input", "inputs", "output", "outputs", "return", "returns", "result", "results",
    "default", "defaults", "note", "notes", "example", "examples",
    "dtype", "tensor", "tensors", "list", "tuple", "type",
    "if", "use", "see", "then", "else",
}
BAD_RETURN_TYPES = {"note", "notes", "example", "examples", "warning", "warnings"}

TYPE_PATTERNS = [
    (r":class:`([^`]+)`", None),  # handled separately
    (r"\btorch\.[A-Za-z_][\w.]*\b", None),
    (r"\btf\.[A-Za-z_][\w.]*\b", None),
    (r"\bjax\.[A-Za-z_][\w.]*\b", None),
    (r"\bpaddle\.[A-Za-z_][\w.]*\b", None),
]

TYPE_CANON = [
    (r"\b(?:str|string|bytes)\b", "str"),
    (r"\b(?:int|integer|long|short)\b", "int"),
    (r"\b(?:float|double|real|half|bfloat16|float16|float32|float64)\b", "float"),
    (r"\bbool(?:ean)?\b", "bool"),
    (r"\btuple\b", "tuple"),
    (r"\blist\b", "list"),
    (r"\bsequence\b", "sequence"),
    (r"\biterable\b", "iterable"),
    (r"\b(?:dict|dictionary|mapping|ordereddict)\b", "dict"),
    (r"\b(?:callable|function|fn)\b", "callable"),
    (r"\b(?:number|numeric|scalar)\b", "number"),
    (r"\b(?:future|futures)\b", "Future"),
    (r"\b(?:tensor|tensors|ndarray|array)\b", "Tensor"),
    (r"\bdevice\b", "device"),
    (r"\bdtype\b", "dtype"),
]

HELPER_FUNCTION_TYPES = {
    "torch.accelerator.current_device_index",
    "torch.cuda.current_device",
}

PROSE_DEFAULT_PATTERNS = [
    r"\bby default\b",
    r"\buses\b",
    r"\bif not given\b",
    r"\bcurrent device\b",
    r"\bcurrent device index\b",
]


def cleanup_type_field(type_str: str) -> str:
    parts = [p.strip() for p in clean_text(type_str).split("|") if p.strip()]
    out: List[str] = []
    seen = set()

    for p in parts:
        if p in HELPER_FUNCTION_TYPES:
            continue
        if p.lower() == "device" and any(x in parts for x in ["torch.device", "str", "int"]):
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)

    return "|".join(out)


def cleanup_default_field(default_str: str) -> str:
    s = clean_text(default_str)
    if not s:
        return ""
    low = s.lower()
    if any(re.search(pat, low) for pat in PROSE_DEFAULT_PATTERNS):
        return "None"
    return s


def cleanup_size_field(size_str: str) -> str:
    s = clean_text(size_str)
    if not s:
        return ""
    bad_words = {"executor", "executors", "created", "different", "wait", "upon"}
    if any(w in s.lower() for w in bad_words):
        return ""
    return s


def repair_output(output: Dict[str, Any]) -> Dict[str, Any]:
    t = clean_text(output.get("type", ""))
    n = clean_text(output.get("numbers", ""))
    d = clean_text(output.get("description", ""))

    low = d.lower()

    if "tuple of two integers" in low:
        t = "tuple[int, int]"
        n = "2"
    elif "named 2-tuple of sets" in low:
        t = "tuple[set, set]"
        n = "2"
    elif ("integer" in low or "bytes" in low) and not n:
        n = "1"

    output["type"] = t
    output["numbers"] = n
    output["description"] = d
    return output

DOC_RELATION_PATTERNS = [
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\s+must\s+have\s+the\s+same\s+shape\b", re.I),
        lambda a, b: f"{a} and {b} must have the same shape",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\s+must\s+have\s+broadcast[- ]compatible\s+shapes\b", re.I),
        lambda a, b: f"{a} and {b} must have broadcast-compatible shapes",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+must\s+have\s+the\s+same\s+(?:dtype|type)\s+as\s+([A-Za-z_]\w*)\b", re.I),
        lambda a, b: f"{a} must have the same dtype as {b}",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+less\s+than\s+([A-Za-z_]\w*)\b", re.I),
        lambda a, b: f"{a} must be less than {b}",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+must\s+be\s+greater\s+than\s+([A-Za-z_]\w*)\b", re.I),
        lambda a, b: f"{a} must be greater than {b}",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+cannot\s+be\s+specified\s+if\s+([A-Za-z_]\w*)\s+is\s+specified\b", re.I),
        lambda a, b: f"{a} cannot be specified with {b}",
    ),
    (
        re.compile(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\s+are\s+mutually\s+exclusive\b", re.I),
        lambda a, b: f"{a} and {b} are mutually exclusive",
    ),
]


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
    return "" if s.lower() in {"none", "null"} else s


def split_lines(doc: str) -> List[str]:
    return str(doc or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def add_unique(target: List[str], value: str) -> None:
    value = clean_text(value)
    if value and value not in target:
        target.append(value)


def normalize_constraint_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        s = clean_text(item)
        if s:
            add_unique(out, s)
    return out


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
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', api_full_name.strip())}.json"


def blank_param_spec() -> Dict[str, Any]:
    return {
        "type": "",
        "size": "",
        "default": "",
        "flag": "",
        "description": "",
        "constraints": [],
    }


def blank_spec() -> Dict[str, Any]:
    return {
        "api_name": "",
        "module_path": "",
        "params": {},
        "output": {"type": "", "numbers": "", "description": ""},
        "constraints": [],
    }


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
            if isinstance(inner, dict):
                return inner
            if isinstance(inner, str):
                candidates.append(inner)
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


def call_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 300,
    temperature: float = 0.0,
    num_predict: int = 320,
    num_ctx: int = 2048,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return str(data["response"])


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


def stringify_default(v: Any) -> str:
    if v is None:
        return "None"
    return clean_text(v)


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
            value = normalize_ws(m.group(1)).strip(",;: ")
            value = value.strip("`")
            if value:
                return value
    return ""


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


def extract_size_text(text: str) -> str:
    s = normalize_ws(text)
    if not s:
        return ""
    patterns = [
        r"\b\d+\s*[- ]?d\b",
        r"\b\d+\s*[- ]?d(?:imension|im)?s?\b",
        r"\bshape\s*=\s*\([^)]*\)",
        r"\bshape\s+\([^)]*\)",
        r"\(\s*[A-Za-z_][\w\s,]*\s*\)",
        r"\btuple of length \d+\b",
        r"\blist of length \d+\b",
        r"\bscalar\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            value = normalize_ws(m.group(0))
            value = value.replace(" -D", "-D").replace("-d", "-D")
            return value
    return ""


def extract_supported_types(type_text: str, desc_text: str = "") -> str:
    text = " ".join(x for x in [clean_text(type_text), clean_text(desc_text)] if x)
    if not text:
        return ""

    out: List[str] = []
    seen = set()

    def add(x: str) -> None:
        x = clean_text(x)
        if x and x.lower() not in {"optional"} and x not in seen:
            seen.add(x)
            out.append(x)

    for m in re.finditer(r":class:`([^`]+)`", text):
        add(m.group(1))

    for pat, _ in TYPE_PATTERNS[1:]:
        for m in re.finditer(pat, text):
            add(m.group(0))

    low = text.lower()
    for pat, label in TYPE_CANON:
        if re.search(pat, low, flags=re.I):
            add(label)

    if out:
        return "|".join(out)

    short = clean_text(type_text)
    if short and len(short.split()) <= 8:
        short = re.sub(r"\boptional\b", "", short, flags=re.I)
        short = normalize_ws(short).strip(", ")
        if short:
            return short

    return ""


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
        if is_suspicious_param_name(name):
            return None
        if allowed_lookup:
            return allowed_lookup.get(name.lower())
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
            size = extract_size_text(desc)
            default = extract_default_from_text(desc)
            if size:
                size_map[name] = size
            if default:
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
            size = extract_size_text(typ)
            default = extract_default_from_text(typ)
            if size and name not in size_map:
                size_map[name] = size
            if default and name not in default_map:
                default_map[name] = default
            continue

        if not in_args:
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
                size = extract_size_text(meta)
                default = extract_default_from_text(meta)
                if size:
                    size_map[name] = size
                if default:
                    default_map[name] = default
            if desc:
                desc_map[name] = desc
                size = extract_size_text(desc)
                default = extract_default_from_text(desc)
                if size:
                    size_map[name] = size
                if default:
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
                size = extract_size_text(meta)
                default = extract_default_from_text(meta)
                if size:
                    size_map[name] = size
                if default:
                    default_map[name] = default
            optional_map[name] = explicit_optional_from_text(meta)
            continue

        if current_name and (line.startswith(" ") or line.startswith("\t")):
            extra = normalize_ws(line)
            if extra:
                prev = desc_map.get(current_name, "")
                desc_map[current_name] = normalize_ws((prev + " " + extra).strip())
                optional_map[current_name] = optional_map.get(current_name, False) or explicit_optional_from_text(extra)
                size = extract_size_text(extra)
                default = extract_default_from_text(extra)
                if size and current_name not in size_map:
                    size_map[current_name] = size
                if default and current_name not in default_map:
                    default_map[current_name] = default

    return type_map, desc_map, size_map, optional_map, default_map


def infer_output_numbers(desc: str) -> str:
    d = (desc or "").lower()
    if not d:
        return ""
    if re.search(r"\b(?:named\s+)?2-?tuple\b|\btuple of two\b|\btwo values\b|\b2 values\b|\breturns 2\b", d):
        return "2"
    if re.search(r"\b(?:named\s+)?3-?tuple\b|\btuple of three\b|\bthree values\b|\b3 values\b|\breturns 3\b", d):
        return "3"
    if re.search(r"\breturns? a tuple\b|\breturns? a list\b", d):
        return "multiple"
    return "1"


def infer_output_type(ret_type: str, ret_desc: str) -> str:
    t = clean_text(ret_type)
    d = clean_text(ret_desc)

    if t:
        if "tuple of two integers" in d.lower():
            return "tuple[int, int]"
        if "named 2-tuple of sets" in d.lower():
            return "tuple[set, set]"
        return t

    low = d.lower()
    if "tuple of two integers" in low:
        return "tuple[int, int]"
    if "named 2-tuple of sets" in low:
        return "tuple[set, set]"
    if "ordered dictionary" in low:
        return "OrderedDict[str, Any]"
    if "dictionary" in low:
        return "dict"
    if "tuple" in low:
        return "tuple"
    if "integer" in low or "bytes" in low:
        return "int"
    return ""


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
            ret_desc_parts.append(normalize_ws(stripped))
        else:
            ret_desc_parts.append(normalize_ws(stripped))

    if not ret_desc_parts:
        m = re.search(r":returns?\s*:\s*(.+)", doc)
        if m:
            ret_desc_parts.append(normalize_ws(m.group(1)))

    ret_desc = normalize_ws(" ".join(ret_desc_parts))
    ret_numbers = infer_output_numbers(ret_desc)
    ret_type = infer_output_type(ret_type, ret_desc)
    return ret_type, ret_numbers, ret_desc


def extract_literal_choices(*texts: str, param_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    blocked = {p.lower() for p in param_names}

    def add(cand: str) -> None:
        cand = clean_text(cand).strip(",.;: ")
        low = cand.lower()
        if not cand:
            return
        if low in blocked or low in BAD_PARAM_NAMES:
            return
        if cand not in seen:
            seen.add(cand)
            out.append(cand)

    for text in texts:
        s = str(text or "")
        if not s:
            continue

        if re.search(r"\b(?:one of|either|options?|values?)\b", s, flags=re.I):
            for cand in re.findall(r"[`'\"]([A-Za-z0-9_.-]+)[`'\"]", s):
                add(cand)
            for cand in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", s):
                add(cand)

    return out


def extract_param_local_constraints(
    name: str,
    type_text: str,
    size_text: str,
    desc_text: str,
    param_names: Sequence[str],
) -> List[str]:
    text = " ".join(x for x in [clean_text(type_text), clean_text(size_text), clean_text(desc_text)] if x)
    low = text.lower()
    out: List[str] = []

    dim_m = re.search(r"\b([1-9])\s*[- ]?d\b", size_text or text, flags=re.I)
    if dim_m:
        add_unique(out, f"{name} must be {dim_m.group(1)}-D")

    if re.search(r"\bnon[- ]negative\b|greater than or equal to 0|>=\s*0", low):
        add_unique(out, f"{name} must be non-negative")
    if re.search(r"\bpositive\b|greater than 0|>\s*0", low) and "less than 0" not in low:
        add_unique(out, f"{name} must be positive")
    if re.search(r"\bnonzero\b|!=\s*0", low):
        add_unique(out, f"{name} must be nonzero")
    if "type must match the current accelerator" in low or "must match the current accelerator device type" in low:
        add_unique(out, f"{name} type must match current accelerator device type")

    for other in param_names:
        if other == name:
            continue
        if re.search(rf"\bsame\s+shape\s+as\s+`?{re.escape(other)}`?\b", text, re.I):
            add_unique(out, f"{name} must have the same shape as {other}")
        if re.search(rf"\bsame\s+(?:dtype|type)\s+as\s+`?{re.escape(other)}`?\b", text, re.I):
            add_unique(out, f"{name} must have the same dtype as {other}")
        if re.search(rf"\bbroadcast[- ]compatible(?:\s+shapes?)?\s+with\s+`?{re.escape(other)}`?\b", text, re.I):
            add_unique(out, f"{name} must be broadcast-compatible with {other}")
        if re.search(rf"\b(?:less\s+than|<)\s+`?{re.escape(other)}`?\b", text, re.I):
            add_unique(out, f"{name} must be less than {other}")
        if re.search(rf"\b(?:greater\s+than|>)\s+`?{re.escape(other)}`?\b", text, re.I):
            add_unique(out, f"{name} must be greater than {other}")

    choices = extract_literal_choices(type_text, desc_text, param_names=param_names)
    if choices:
        add_unique(out, f"{name} must be one of {', '.join(choices)}")

    return out


def extract_parameter_constraints_and_globals(
    doc: str,
    params: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    param_names = list(params.keys())
    param_constraints: Dict[str, List[str]] = {p: [] for p in param_names}
    global_constraints: List[str] = []

    for p, meta in params.items():
        for rule in normalize_constraint_list(meta.get("constraints", [])):
            add_unique(param_constraints[p], rule)
        for rule in extract_param_local_constraints(
            name=p,
            type_text=meta.get("type", ""),
            size_text=meta.get("size", ""),
            desc_text=meta.get("description", ""),
            param_names=param_names,
        ):
            add_unique(param_constraints[p], rule)

    text = "\n".join(normalize_ws(ln) for ln in split_lines(doc) if normalize_ws(ln))
    lookup = {p.lower(): p for p in param_names}

    for rx, builder in DOC_RELATION_PATTERNS:
        for m in rx.finditer(text):
            groups = [g for g in m.groups() if isinstance(g, str)]
            resolved: List[str] = []
            ok = True
            for g in groups:
                actual = lookup.get(g.lower())
                if not actual:
                    ok = False
                    break
                resolved.append(actual)
            if not ok:
                continue
            rule = builder(*resolved)
            for name in sorted(set(resolved)):
                add_unique(param_constraints[name], rule)
            if len(set(resolved)) > 1:
                add_unique(global_constraints, rule)

    axis_name = next((p for p in param_names if p.lower() in {"axis", "dim", "dims", "dimension"}), "")
    input_name = next((p for p in param_names if p.lower() in {"input", "x", "tensor", "values", "params", "flat_values"}), "")
    low = text.lower()
    if axis_name and input_name and re.search(r"\b(rank|ndim|dimensions)\b", low):
        add_unique(param_constraints[axis_name], f"{axis_name} must satisfy -rank({input_name}) <= {axis_name} < rank({input_name})")

    return param_constraints, global_constraints


def param_mentions(text: str, param_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in param_names:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.I):
            out.append(p)
    return out


def redistribute_global_constraints(
    raw_constraints: Any,
    param_names: Sequence[str],
    param_constraints: Dict[str, List[str]],
    global_constraints: List[str],
) -> None:
    for rule in normalize_constraint_list(raw_constraints):
        mentioned = param_mentions(rule, param_names)
        if not mentioned:
            add_unique(global_constraints, rule)
            continue
        for p in mentioned:
            add_unique(param_constraints[p], rule)
        if len(mentioned) > 1:
            add_unique(global_constraints, rule)

def compact_doc_for_prompt(doc: str) -> str:
    lines = split_lines(doc)
    kept: List[str] = []

    in_args = False
    in_returns = False

    keywords = (
        "default", "optional", "must", "one of", "either",
        "dtype", "shape", "rank", "dimension", "tuple", "returns"
    )

    for line in lines:
        stripped = line.strip()
        lower = stripped.rstrip(":").lower()

        if not stripped:
            continue

        if "(" in stripped and ")" in stripped and not stripped.endswith(":"):
            kept.append(stripped)
            continue

        if lower in SECTION_ARGS:
            in_args = True
            in_returns = False
            kept.append(stripped)
            continue

        if lower in SECTION_RETURNS:
            in_args = False
            in_returns = True
            kept.append(stripped)
            continue

        if lower in STOP_SECTIONS:
            in_args = False
            in_returns = False
            continue

        if in_args or in_returns:
            kept.append(line)
            continue

        if any(k in lower for k in keywords):
            kept.append(line)

    return "\n".join(kept[:250])

def build_prompt(api_full_name: str, doc: str) -> str:
    sig_line = find_signature_line(doc)
    sig_params, _, _ = parse_signature_params(sig_line)

    if sig_params:
        supported = sig_params
    else:
        doc_type_map, doc_desc_map, _, _, _ = parse_doc_params(doc, allowed_names=None)
        supported = []
        seen = set()
        for p in list(doc_type_map.keys()) + list(doc_desc_map.keys()):
            if is_suspicious_param_name(p):
                continue
            if p not in seen:
                supported.append(p)
                seen.add(p)

    supported_text = ", ".join(supported) if supported else "(none reliably parsed)"
    doc_for_prompt = compact_doc_for_prompt(doc)

    return PROMPT_TEMPLATE.format(
        api_full_name=api_full_name,
        sig_line=sig_line or "",
        supported_params=supported_text,
        api_doc_text=doc_for_prompt,
    )


def normalize_schema(spec: Dict[str, Any], api_full_name: str, doc: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        spec = blank_spec()

    module_path, _, api_name = api_full_name.rpartition(".")
    sig_line = find_signature_line(doc)
    sig_params, sig_defaults, sig_ret = parse_signature_params(sig_line)

    doc_type_map, doc_desc_map, doc_size_map, doc_optional_map, doc_default_map = parse_doc_params(
        doc,
        allowed_names=sig_params if sig_params else None,
    )
    ret_type_doc, ret_numbers_doc, ret_desc_doc = parse_return_info(doc)

    out = blank_spec()
    out["api_name"] = api_name or api_full_name
    out["module_path"] = module_path

    llm_params = normalize_llm_params(spec.get("params"))

    if sig_params:
        supported_names = list(sig_params)
    else:
        supported_names = []
        seen = set()
        for p in list(doc_type_map.keys()) + list(doc_desc_map.keys()) + list(llm_params.keys()):
            if is_suspicious_param_name(p):
                continue
            if p and p not in seen:
                supported_names.append(p)
                seen.add(p)

    final_params: Dict[str, Dict[str, Any]] = {}
    for name in supported_names:
        llm_p = llm_params.get(name, {}) if isinstance(llm_params.get(name, {}), dict) else {}

        raw_type_text = clean_text(doc_type_map.get(name, "")) or clean_text(llm_p.get("type", ""))
        desc = clean_text(doc_desc_map.get(name, "")) or clean_text(llm_p.get("description", ""))
        size = clean_text(doc_size_map.get(name, "")) or extract_size_text(clean_text(llm_p.get("size", "")))
        inferred_type = extract_supported_types(raw_type_text, desc)

        default = ""
        if name in sig_defaults:
            default = stringify_default(sig_defaults[name])
        elif name in doc_default_map:
            default = doc_default_map[name]
        else:
            default = clean_text(llm_p.get("default", ""))

        if name in sig_defaults:
            flag = "Optional"
        elif doc_optional_map.get(name, False):
            flag = "Optional"
        elif clean_text(llm_p.get("flag", "")) == "Optional":
            flag = "Optional"
        elif name in sig_params or supported_names:
            flag = "Required"
        else:
            flag = ""

        param_obj = blank_param_spec()
        param_obj.update(
            {
                "type": inferred_type,
                "size": size,
                "default": default,
                "flag": flag,
                "description": desc,
                "constraints": normalize_constraint_list(llm_p.get("constraints")),
            }
        )
        final_params[name] = param_obj

    raw_output = spec.get("output") if isinstance(spec.get("output"), dict) else {}
    output_type = clean_text(raw_output.get("type", "")) or ret_type_doc or clean_text(sig_ret or "")
    out["output"] = {
        "type": output_type,
        "numbers": clean_text(raw_output.get("numbers", "")) or ret_numbers_doc,
        "description": ret_desc_doc or clean_text(raw_output.get("description", "")),
    }

    param_constraints, global_constraints = extract_parameter_constraints_and_globals(doc, final_params)
    redistribute_global_constraints(
        raw_constraints=spec.get("constraints"),
        param_names=list(final_params.keys()),
        param_constraints=param_constraints,
        global_constraints=global_constraints,
    )

    for name in final_params:
        final_params[name]["constraints"] = param_constraints.get(name, [])

    # deterministic cleanup
    for name, meta in final_params.items():
        meta["type"] = cleanup_type_field(meta.get("type", ""))
        meta["default"] = cleanup_default_field(meta.get("default", ""))
        meta["size"] = cleanup_size_field(meta.get("size", ""))

    # targeted fix for futures.wait
    if api_full_name.endswith(".wait") and "return_when" in final_params:
        add_unique(
            final_params["return_when"]["constraints"],
            "return_when must be one of FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED",
        )
        if not final_params["return_when"]["type"]:
            final_params["return_when"]["type"] = "str"

    if api_full_name.endswith(".wait") and "fs" in final_params:
        final_params["fs"]["size"] = ""
        if not final_params["fs"]["type"]:
            final_params["fs"]["type"] = "sequence|Future"

    out["output"] = repair_output(out["output"])
    out["params"] = final_params
    out["constraints"] = global_constraints
    return out

def validate_spec(spec: Dict[str, Any], api_full_name: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError("Spec is not a dict")

    if clean_text(spec.get("api_name", "")) == "":
        raise ValueError(f"{api_full_name}: api_name is empty")

    if "params" not in spec or not isinstance(spec["params"], dict):
        raise ValueError(f"{api_full_name}: params is missing or not a dict")

    if "output" not in spec or not isinstance(spec["output"], dict):
        raise ValueError(f"{api_full_name}: output is missing or not a dict")

    for name, meta in spec["params"].items():
        if is_suspicious_param_name(name):
            raise ValueError(f"{api_full_name}: suspicious param name {name}")
        if not isinstance(meta, dict):
            raise ValueError(f"{api_full_name}: param {name} is not a dict")

        if clean_text(meta.get("flag", "")) not in {"", "Required", "Optional"}:
            raise ValueError(f"{api_full_name}: invalid flag for {name}")

        t = clean_text(meta.get("type", ""))
        if any(x in t for x in HELPER_FUNCTION_TYPES):
            raise ValueError(f"{api_full_name}: helper function leaked into type for {name}")

        dflt = clean_text(meta.get("default", ""))
        if dflt and "default" in dflt.lower() and dflt != "None":
            raise ValueError(f"{api_full_name}: prose default leaked into default field for {name}")

        sz = clean_text(meta.get("size", ""))
        if sz and len(sz.split()) > 6:
            raise ValueError(f"{api_full_name}: size field looks like prose for {name}")

    out_t = clean_text(spec["output"].get("type", ""))
    out_d = clean_text(spec["output"].get("description", "")).lower()
    if out_t == "string" and "bytes" in out_d:
        raise ValueError(f"{api_full_name}: output type likely wrong")

def process_one(
    api_full_name: str,
    api_doc_text: str,
    outdir: str,
    model: str,
    host: str,
    timeout: int,
    retries: int,
    sleep: float,
    num_predict: int,
    num_ctx: int,
    temperature: float,
) -> Tuple[bool, str]:
    prompt = build_prompt(api_full_name=api_full_name, doc=api_doc_text)
    last_err = ""

    for attempt in range(1, retries + 1):
        try:
            raw = call_ollama(
                prompt,
                model=model,
                host=host,
                timeout=timeout,
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
            )
            parsed = extract_json(raw)
            normalized = normalize_schema(parsed, api_full_name, api_doc_text)
            validate_spec(normalized, api_full_name)
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
    return {"processed": 0, "succeeded": 0, "failed": 0, "failures": []}


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
    num_predict: int = 320,
    num_ctx: int = 2048,
    temperature: float = 0.0,
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
            num_predict=num_predict,
            num_ctx=num_ctx,
            temperature=temperature,
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
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": temperature,
        })
        write_json(summary_path, summary)

    print(
        f"[done] total={len(items)} processed={summary.get('processed', 0)} "
        f"succeeded={summary.get('succeeded', 0)} failed={summary.get('failed', 0)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert API docs into Stage-2 JSON specs using Ollama with deterministic normalization"
    )
    ap.add_argument("--input", required=True, help="CSV/XLSX with api_full_name, api_doc_text")
    ap.add_argument("--outdir", required=True, help="Output directory for per-API JSON files")
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--combined-out", default="", help="Optional combined JSON list path")
    ap.add_argument("--num-predict", type=int, default=320, help="Ollama num_predict")
    ap.add_argument("--num-ctx", type=int, default=2048, help="Ollama num_ctx")
    ap.add_argument("--temperature", type=float, default=0.0, help="Ollama temperature")
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
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())