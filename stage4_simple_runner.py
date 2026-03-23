from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import math
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SKIP_FILENAMES = {
    "summary.json",
    "generation_summary.json",
    "init2test_summary.json",
    "mutation_summary.json",
}

MODULE_ALIASES = {
    "tf": "tensorflow",
    "np": "numpy",
    "pd": "paddle",
    "jnp": "jax.numpy",
}

DTYPE_RE = re.compile(
    r"\b("
    r"bfloat16|half|float16|float32|float64|"
    r"complex64|complex128|"
    r"uint8|uint16|uint32|uint64|"
    r"int8|int16|int32|int64|"
    r"bool|string"
    r")\b",
    re.I,
)

FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}
INT_DTYPES = {"int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64"}
COMPLEX_DTYPES = {"complex64", "complex128"}

PREFERRED_DTYPES = [
    "float32",
    "int32",
    "float64",
    "int64",
    "bool",
    "bfloat16",
    "float16",
    "complex64",
    "complex128",
    "string",
]

SYMBOLIC_DIMS = {
    "n": 2,
    "b": 2,
    "batch": 2,
    "c": 3,
    "h": 8,
    "w": 8,
    "d": 4,
    "m": 2,
    "k": 2,
    "r": 2,
    "rank": 2,
    "rows": 2,
    "cols": 2,
}


@dataclass
class ParamSpec:
    type: str = ""
    size: str = ""
    default: str = ""
    flag: str = ""
    description: str = ""
    constraints: List[str] = field(default_factory=list)


@dataclass
class APISpec:
    api_name: str
    module_path: str
    params: Dict[str, ParamSpec]
    output: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    source_file: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.module_path}.{self.api_name}" if self.module_path else self.api_name


@dataclass
class RunResult:
    success: bool
    runtime_ms: float
    output_summary: Dict[str, Any]
    has_nan_or_inf: bool
    exception_type: str = ""
    exception_message: str = ""
    traceback_text: str = ""
    inconsistency: bool = False
    inconsistency_detail: str = ""


@dataclass
class MutationRecord:
    step: int
    parameter: str
    family: str
    strategy: str
    likely_valid: bool
    before: Any
    after: Any


class MutationHistory:
    def __init__(self) -> None:
        self.failed_fingerprints: set[str] = set()
        self.family_stats: Dict[str, Counter] = defaultdict(Counter)

    def fingerprint(self, value: Any) -> str:
        return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)

    def seen_bad(self, value: Any) -> bool:
        return self.fingerprint(value) in self.failed_fingerprints

    def record(self, family: str, success: bool, value: Any) -> None:
        self.family_stats[family]["success" if success else "failure"] += 1
        if not success:
            self.failed_fingerprints.add(self.fingerprint(value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def parse_default(text: str) -> Any:
    text = normalize_text(text)
    if not text:
        return None
    low = text.lower()
    if low in {"none", "null"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import ast
        return ast.literal_eval(text)
    except Exception:
        return text


def normalize_module_path(path: str) -> str:
    if not path:
        return path
    parts = path.split(".")
    parts[0] = MODULE_ALIASES.get(parts[0], parts[0])
    return ".".join(parts)


def detect_framework(spec: APISpec) -> str:
    root = normalize_module_path(spec.module_path).split(".")[
        0] if spec.module_path else "python"
    if root in {"tensorflow", "torch", "jax", "numpy", "paddle"}:
        return root
    return "python"


def load_api_specs(spec_dir: str) -> Dict[str, APISpec]:
    specs: Dict[str, APISpec] = {}
    for path in sorted(Path(spec_dir).glob("*.json")):
        if path.name in SKIP_FILENAMES:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            params: Dict[str, ParamSpec] = {}
            for name, info in (entry.get("params") or {}).items():
                info = info or {}
                params[name] = ParamSpec(
                    type=str(info.get("type", "") or ""),
                    size=str(info.get("size", "") or ""),
                    default=str(info.get("default", "") or ""),
                    flag=str(info.get("flag", "") or ""),
                    description=str(info.get("description", "") or ""),
                    constraints=list(info.get("constraints", []) or []),
                )
            spec = APISpec(
                api_name=str(entry.get("api_name", "") or ""),
                module_path=str(entry.get("module_path", "") or ""),
                params=params,
                output=dict(entry.get("output", {}) or {}),
                constraints=list(entry.get("constraints", []) or []),
                source_file=str(path),
            )
            if spec.full_name:
                specs[spec.full_name] = spec
    return specs


def allowed_dtypes(*texts: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for text in texts:
        for token in DTYPE_RE.findall(text or ""):
            dt = token.lower()
            if dt == "half":
                dt = "float16"
            if dt not in seen:
                seen.add(dt)
                out.append(dt)
    return out


def choose_dtype(candidates: Sequence[str]) -> str:
    if not candidates:
        return "float32"
    for dt in PREFERRED_DTYPES:
        if dt in candidates:
            return dt
    return candidates[0]


def looks_like_dtype_param(name: str, param: ParamSpec) -> bool:
    low_name = name.lower()
    text = normalize_text(
        f"{param.type} {param.description} {' '.join(param.constraints)}").lower()
    if low_name in {"dtype", "dtypes", "output_type", "out_type"}:
        return True
    if low_name.endswith("_dtype") or low_name.startswith("dtype_"):
        return True
    return "dtype" in text and "tensor" not in text


def is_tensor_like(param: ParamSpec) -> bool:
    text = normalize_text(
        f"{param.type} {param.size} {param.description} {' '.join(param.constraints)}").lower()
    return any(token in text for token in [
        "tensor", "ndarray", "array-like", "array", "raggedtensor", "sparsetensor"
    ])


def is_list_of_tensors(name: str, param: ParamSpec) -> bool:
    text = normalize_text(
        f"{param.type} {param.description} {' '.join(param.constraints)}").lower()
    return (
        name.lower() in {"inputs", "tensors"} and "tensor" in text
    ) or (
        any(token in text for token in [
            "list", "tuple", "sequence", "iterable"]) and "tensor" in text
    )


def is_required(param: ParamSpec) -> bool:
    return normalize_text(param.flag).lower() == "required"


def _parse_number(text: str) -> Optional[float]:
    low = normalize_text(text).lower()
    if low in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if low in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(low)
    except Exception:
        return None


def parse_numeric_range(*texts: str) -> Optional[Dict[str, Any]]:
    info: Dict[str, Any] = {}
    for text in texts:
        text = normalize_text(text)
        if not text:
            continue

        m = re.search(
            r"(?:input\s+)?range(?:\s+for\s+[A-Za-z_]\w*)?\s*(?:is)?\s*([\[(])\s*([^,\])]+)\s*,\s*([^\])]+)\s*([\])])", text, re.I)
        if m:
            left, lo_raw, hi_raw, right = m.groups()
            lo = _parse_number(lo_raw)
            hi = _parse_number(hi_raw)
            if lo is not None:
                info["min"] = lo
                info["min_inclusive"] = left == "["
            if hi is not None:
                info["max"] = hi
                info["max_inclusive"] = right == "]"

        for pat, key, inclusive in [
            (r"must be\s*>=\s*([^ ,;]+)", "min", True),
            (r"must be\s*>\s*([^ ,;]+)", "min", False),
            (r"must be\s*<=\s*([^ ,;]+)", "max", True),
            (r"must be\s*<\s*([^ ,;]+)", "max", False),
        ]:
            m = re.search(pat, text, re.I)
            if m:
                num = _parse_number(m.group(1))
                if num is not None:
                    info[key] = num
                    info[f"{key}_inclusive"] = inclusive

        if "nonnegative" in text.lower():
            info["min"] = 0.0
            info["min_inclusive"] = True

        if re.search(r"positive", text, re.I) and "nonpositive" not in text.lower():
            info.setdefault("min", 0.0)
            info.setdefault("min_inclusive", False)

        if re.search(r"negative", text, re.I) and "nonnegative" not in text.lower():
            info.setdefault("max", 0.0)
            info.setdefault("max_inclusive", False)

    return info or None


def bound_scalar(value: float, bounds: Optional[Dict[str, Any]]) -> float:
    if not bounds:
        return value
    if "min" in bounds and not math.isinf(bounds["min"]):
        min_val = bounds["min"]
        if bounds.get("min_inclusive", True):
            value = max(value, min_val)
        else:
            value = max(value, min_val + 1e-3)
    if "max" in bounds and not math.isinf(bounds["max"]):
        max_val = bounds["max"]
        if bounds.get("max_inclusive", True):
            value = min(value, max_val)
        else:
            value = min(value, max_val - 1e-3)
    return value


def parse_dims(*texts: str) -> Optional[List[int]]:
    for text in texts:
        text = normalize_text(text)
        if not text:
            continue
        m = re.search(r"\[([^\]]+)\]", text)
        if m:
            dims = _parse_dim_tokens(m.group(1))
            if dims:
                return dims
        for m in re.finditer(r"\(([^()]*)\)", text):
            dims = _parse_dim_tokens(m.group(1))
            if dims:
                return dims
        m = re.search(r"\b(\d+(?:x\d+)+)\b", text.lower())
        if m:
            return [int(x) for x in m.group(1).split("x")]
        if re.search(r"\b1d\b|\b1-d\b", text, re.I):
            return [2]
        if re.search(r"\b2d\b|\b2-d\b", text, re.I):
            return [2, 2]
        if re.search(r"\b3d\b|\b3-d\b", text, re.I):
            return [2, 2, 2]
        if re.search(r"\b4d\b|\b4-d\b", text, re.I):
            return [1, 3, 8, 8]
    return None


def _parse_dim_tokens(text: str) -> Optional[List[int]]:
    dims: List[int] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            dims.append(max(1, int(tok)))
        elif re.fullmatch(r"[a-z_][a-z0-9_]*", tok):
            dims.append(SYMBOLIC_DIMS.get(tok, 2))
        else:
            return None
    return dims or None


def _nested_fill(shape: Sequence[int], value: Any) -> Any:
    if not shape:
        return value
    return [_nested_fill(shape[1:], value) for _ in range(shape[0])]


def _encode_complex(arr: np.ndarray) -> Any:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return {"real": float(arr.real), "imag": float(arr.imag)}
    return [_encode_complex(x) for x in arr]


def _decode_complex(data: Any) -> Any:
    if isinstance(data, dict) and set(data) == {"real", "imag"}:
        return complex(data["real"], data["imag"])
    if isinstance(data, list):
        return [_decode_complex(x) for x in data]
    return data


def make_tensor_payload(framework: str, dtype: str, shape: Sequence[int], bounds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rng = np.random.default_rng()
    shape = list(shape)

    if dtype == "string":
        data = _nested_fill(shape, "x")
    elif dtype == "bool":
        data = rng.integers(0, 2, size=shape).astype(np.bool_).tolist()
    elif dtype in INT_DTYPES:
        lo = 0
        hi = 4
        if bounds:
            if "min" in bounds and not math.isinf(bounds["min"]):
                lo = int(math.ceil(bounds["min"] if bounds.get(
                    "min_inclusive", True) else bounds["min"] + 1))
            if "max" in bounds and not math.isinf(bounds["max"]):
                hi = int(math.floor(bounds["max"] if bounds.get(
                    "max_inclusive", True) else bounds["max"] - 1)) + 1
        if hi <= lo:
            hi = lo + 2
        data = rng.integers(lo, hi, size=shape).tolist()
    elif dtype in COMPLEX_DTYPES:
        lo = -1.0
        hi = 1.0
        if bounds:
            lo = bound_scalar(lo, {"min": bounds.get(
                "min", lo), "min_inclusive": bounds.get("min_inclusive", True)})
            hi = bound_scalar(hi, {"max": bounds.get(
                "max", hi), "max_inclusive": bounds.get("max_inclusive", True)})
            if hi <= lo:
                hi = lo + 1.0
        reals = rng.uniform(lo, hi, size=shape)
        imags = rng.uniform(-1.0, 1.0, size=shape)
        data = _encode_complex(reals + 1j * imags)
    else:
        lo = -1.0
        hi = 1.0
        if bounds:
            if "min" in bounds and not math.isinf(bounds["min"]):
                lo = bounds["min"] if bounds.get(
                    "min_inclusive", True) else bounds["min"] + 1e-3
            if "max" in bounds and not math.isinf(bounds["max"]):
                hi = bounds["max"] if bounds.get(
                    "max_inclusive", True) else bounds["max"] - 1e-3
        if hi <= lo:
            hi = lo + 1.0
        data = rng.uniform(lo, hi, size=shape).astype(np.float32).tolist()

    return {
        "kind": "tensor",
        "framework": framework,
        "dtype": dtype,
        "shape": shape,
        "data": data,
    }


def choose_scalar(name: str, param: ParamSpec, ctx: Dict[str, Any]) -> Any:
    low = name.lower()
    text = normalize_text(
        f"{param.type} {param.description} {' '.join(param.constraints)}").lower()
    bounds = parse_numeric_range(param.description, *param.constraints)

    if low in {"shape", "dims", "sizes"}:
        if ctx.get("primary_tensor_shape"):
            return list(ctx["primary_tensor_shape"])
        if ctx.get("receiver_shape"):
            return list(ctx["receiver_shape"])
        return [2, 2]

    if low in {"rank", "ndims"}:
        if ctx.get("receiver_shape"):
            return len(ctx["receiver_shape"])
        if ctx.get("primary_tensor_shape"):
            return len(ctx["primary_tensor_shape"])
        return 2

    if low in {"axis", "dim"}:
        rank = len(ctx.get("primary_tensor_shape")
                   or ctx.get("receiver_shape") or [1])
        return 0 if rank == 0 else min(rank - 1, 0)

    if low in {"name"}:
        return "seed"

    if looks_like_dtype_param(name, param):
        return choose_dtype(allowed_dtypes(param.type, param.description, *param.constraints) or [ctx.get("primary_tensor_dtype", "float32")])

    if "bool" in text:
        return True

    if "string" in text or re.fullmatch(r"str(ing)?", normalize_text(param.type).lower()):
        return "x"

    if low in {"row_splits", "nested_row_splits"}:
        splits = [0, 1, 3]
        return splits if "nested" not in low else [[0, 2], [0, 1, 3]]

    if low in {"row_lengths", "nested_row_lengths"}:
        lengths = [1, 2]
        return lengths if "nested" not in low else [[2], [1, 1]]

    if low in {"value_rowids", "rowids", "nested_value_rowids"}:
        rowids = [0, 1, 1]
        return rowids if "nested" not in low else [[0, 0], [0, 1, 1]]

    if "float" in text or low in {"alpha", "beta", "eps", "epsilon"}:
        value = 0.25
        if bounds and "min" in bounds and not math.isinf(bounds["min"]):
            value = bounds["min"] if bounds.get(
                "min_inclusive", True) else bounds["min"] + 1e-3
        return bound_scalar(float(value), bounds)

    if "int" in text or low in {"k", "num", "count", "depth", "block_size", "blocksize"}:
        if bounds and "min" in bounds and not math.isinf(bounds["min"]):
            value = int(math.ceil(bounds["min"] if bounds.get(
                "min_inclusive", True) else bounds["min"] + 1))
            return max(0, value)
        return 1

    if any(token in text for token in ["tuple", "list", "sequence", "iterable"]):
        return [1]

    return 1


def infer_tensor_dtype(name: str, param: ParamSpec, ctx: Dict[str, Any]) -> str:
    low = name.lower()
    allowed = allowed_dtypes(param.type, param.description, *param.constraints)
    if low in {"indices", "row_splits", "row_lengths", "value_rowids", "rowids", "nested_row_splits", "nested_row_lengths", "nested_value_rowids"}:
        ints = [x for x in allowed if x in INT_DTYPES]
        return choose_dtype(ints or ["int32"])
    if low == "mask":
        return "bool"
    if low == "value" and "scalar" in normalize_text(f"{param.type} {param.description} {' '.join(param.constraints)}").lower():
        return ctx.get("primary_tensor_dtype", choose_dtype(allowed or ["float32"]))
    return choose_dtype(allowed or [ctx.get("primary_tensor_dtype", "float32")])


def make_receiver_payload(spec: APISpec) -> Optional[Dict[str, Any]]:
    framework = detect_framework(spec)
    module = normalize_module_path(spec.module_path)
    tail = module.split(".")[-1] if module else ""

    # Class/static constructors like RaggedTensor.from_row_splits should not materialize a receiver.
    if spec.api_name.startswith("from_"):
        return None

    if framework == "tensorflow":
        if tail == "Variable":
            return {
                "kind": "variable",
                "framework": framework,
                "dtype": "float32",
                "shape": [3, 3],
                "data": [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            }
        if tail == "RaggedTensor":
            return {"kind": "ragged_tensor", "framework": framework, "dtype": "float32", "data": [[1.0], [2.0, 3.0]]}
        if tail == "SparseTensor":
            return {
                "kind": "sparse_tensor",
                "framework": framework,
                "dtype": "float32",
                "indices": [[0, 0], [1, 1]],
                "values": [1.0, 2.0],
                "dense_shape": [2, 2],
            }
        if tail == "TensorArray":
            return {
                "kind": "tensor_array",
                "framework": framework,
                "dtype": "float32",
                "size": 2,
                "elements": [[1.0, 2.0], [3.0, 4.0]],
            }
        if tail == "TensorShape":
            return {"kind": "tensor_shape", "framework": framework, "dims": [2, 2]}
        if tail == "TensorSpec":
            return {"kind": "tensor_spec", "framework": framework, "shape": [2, 2], "dtype": "float32"}
        if tail == "DType":
            return {"kind": "dtype_obj", "framework": framework, "value": "float32"}

    if framework == "torch" and tail.endswith("Tensor"):
        dtype = "float32"
        low = tail.lower()
        if "bfloat16" in low:
            dtype = "bfloat16"
        elif "float16" in low or "half" in low:
            dtype = "float16"
        elif "double" in low or "float64" in low:
            dtype = "float64"
        elif "int64" in low or "long" in low:
            dtype = "int64"
        elif "int32" in low:
            dtype = "int32"
        return make_tensor_payload(framework, dtype, [2, 2])

    if framework == "jax" and (tail.endswith("Array") or tail.endswith("Tensor")):
        return make_tensor_payload(framework, "float32", [2, 2])

    return None


def maybe_tensor_shape(name: str, param: ParamSpec, spec: APISpec, ctx: Dict[str, Any]) -> List[int]:
    dims = parse_dims(param.size, param.description, *param.constraints)
    if dims is not None:
        return dims

    low = name.lower()
    text = normalize_text(
        f"{param.description} {' '.join(param.constraints)}").lower()

    if low == "mask":
        return list(ctx.get("primary_tensor_shape") or ctx.get("receiver_shape") or [2, 2])

    if low == "value" and "same shape" in text:
        return list(ctx.get("primary_tensor_shape") or ctx.get("receiver_shape") or [2, 2])

    if low == "indices":
        receiver_shape = ctx.get("receiver_shape") or [2, 2]
        rank = len(receiver_shape)
        return [2, min(2, max(1, rank))]

    if low in {"x", "input", "tensor"} and len(spec.params) == 1:
        return []

    if low in {"x", "input", "tensor", "value", "values"} and "same shape" not in text:
        return [2, 2]

    if low in {"shape", "dims"}:
        return [2]

    return [2, 2]


def _build_special_param(name: str, param: ParamSpec, framework: str, ctx: Dict[str, Any]) -> Tuple[bool, Any]:
    low = name.lower()

    if low in {"row_splits", "nested_row_splits"}:
        if low == "row_splits":
            return True, {"kind": "tensor", "framework": framework, "dtype": "int32", "shape": [3], "data": [0, 1, 3]}
        return True, [
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [2], "data": [0, 2]},
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [3], "data": [0, 1, 3]},
        ]

    if low in {"row_lengths", "nested_row_lengths"}:
        if low == "row_lengths":
            return True, {"kind": "tensor", "framework": framework, "dtype": "int32", "shape": [2], "data": [1, 2]}
        return True, [
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [1], "data": [2]},
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [2], "data": [1, 1]},
        ]

    if low in {"value_rowids", "nested_value_rowids", "rowids"}:
        if low in {"value_rowids", "rowids"}:
            return True, {"kind": "tensor", "framework": framework, "dtype": "int32", "shape": [3], "data": [0, 1, 1]}
        return True, [
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [2], "data": [0, 0]},
            {"kind": "tensor", "framework": framework,
                "dtype": "int32", "shape": [3], "data": [0, 1, 1]},
        ]

    if low == "mask":
        shape = list(ctx.get("primary_tensor_shape")
                     or ctx.get("receiver_shape") or [2, 2])
        return True, {"kind": "tensor", "framework": framework, "dtype": "bool", "shape": shape, "data": _nested_fill(shape, True)}

    if low == "value" and "scalar" in normalize_text(f"{param.type} {param.description} {' '.join(param.constraints)}").lower():
        return True, 1.0 if ctx.get("primary_tensor_dtype", "float32") in FLOAT_DTYPES else 1

    if low in {"shape", "dims", "sizes"}:
        return True, list(ctx.get("primary_tensor_shape") or ctx.get("receiver_shape") or [2, 2])

    if low in {"other", "target_dtype", "src_type", "dst_type"} and looks_like_dtype_param(name, param):
        return True, choose_dtype(allowed_dtypes(param.type, param.description, *param.constraints) or [ctx.get("primary_tensor_dtype", "float32")])

    return False, None


def update_context_from_value(name: str, value: Any, ctx: Dict[str, Any]) -> None:
    low = name.lower()
    if isinstance(value, dict) and value.get("kind") in {"tensor", "variable"}:
        ctx[f"{low}_shape"] = list(value.get("shape", []))
        ctx[f"{low}_dtype"] = value.get("dtype", "float32")
        if low in {"x", "input", "tensor", "values"} and "primary_tensor_shape" not in ctx:
            ctx["primary_tensor_shape"] = list(value.get("shape", []))
            ctx["primary_tensor_dtype"] = value.get("dtype", "float32")


def generate_seed_case(spec: APISpec, include_optional: bool = False) -> Dict[str, Any]:
    framework = detect_framework(spec)
    case: Dict[str, Any] = {
        "api": spec.full_name,
        "framework": framework,
        "module_path": normalize_module_path(spec.module_path),
        "parameters": {},
    }

    receiver = make_receiver_payload(spec)
    if receiver is not None:
        case["receiver"] = receiver

    ctx: Dict[str, Any] = {}
    if isinstance(receiver, dict):
        if receiver.get("kind") in {"tensor", "variable"}:
            ctx["receiver_shape"] = list(receiver.get("shape", []))
            ctx["primary_tensor_shape"] = list(receiver.get("shape", []))
            ctx["primary_tensor_dtype"] = receiver.get("dtype", "float32")
        elif receiver.get("kind") == "tensor_shape":
            ctx["receiver_shape"] = list(receiver.get("dims", []))
        elif receiver.get("kind") == "tensor_spec":
            ctx["receiver_shape"] = list(receiver.get("shape", []))
            ctx["primary_tensor_shape"] = list(receiver.get("shape", []))
            ctx["primary_tensor_dtype"] = receiver.get("dtype", "float32")

    for name, param in spec.params.items():
        default = parse_default(param.default)
        required = is_required(param)
        if not required and not include_optional and default is None:
            continue

        if default is not None and default != "":
            case["parameters"][name] = default
            update_context_from_value(name, default, ctx)
            continue

        handled, special = _build_special_param(name, param, framework, ctx)
        if handled:
            case["parameters"][name] = special
            update_context_from_value(name, special, ctx)
            continue

        if looks_like_dtype_param(name, param):
            value = choose_dtype(allowed_dtypes(param.type, param.description,
                                 *param.constraints) or [ctx.get("primary_tensor_dtype", "float32")])
            case["parameters"][name] = value
            continue

        if is_list_of_tensors(name, param):
            dtype = infer_tensor_dtype(name, param, ctx)
            shape = maybe_tensor_shape(name, param, spec, ctx)
            bounds = parse_numeric_range(param.description, *param.constraints)
            value = [
                make_tensor_payload(framework, dtype, shape, bounds),
                make_tensor_payload(framework, dtype, shape, bounds),
            ]
            case["parameters"][name] = value
            continue

        if is_tensor_like(param):
            dtype = infer_tensor_dtype(name, param, ctx)
            bounds = parse_numeric_range(param.description, *param.constraints)
            shape = maybe_tensor_shape(name, param, spec, ctx)

            if name.lower() == "indices" and ctx.get("receiver_shape"):
                rank = len(ctx["receiver_shape"])
                width = min(2, max(1, rank))
                rows = [[0] * width]
                if ctx["receiver_shape"] and ctx["receiver_shape"][0] > 1:
                    rows.append([1] + [0] * (width - 1))
                value = {
                    "kind": "tensor",
                    "framework": framework,
                    "dtype": "int32",
                    "shape": [len(rows), width],
                    "data": rows,
                }
            else:
                value = make_tensor_payload(framework, dtype, shape, bounds)

            case["parameters"][name] = value
            update_context_from_value(name, value, ctx)
            continue

        value = choose_scalar(name, param, ctx)
        case["parameters"][name] = value

    return case


def infer_data_shape(data: Any) -> Optional[List[int]]:
    if isinstance(data, dict) and set(data) == {"real", "imag"}:
        return []
    if not isinstance(data, list):
        return []
    if not data:
        return [0]
    shapes = [infer_data_shape(x) for x in data]
    if any(s is None for s in shapes):
        return None
    first = shapes[0]
    if any(s != first for s in shapes[1:]):
        return None
    return [len(data)] + first


def validate_case(spec: APISpec, case: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    params = case.get("parameters", {})

    for name, param in spec.params.items():
        if is_required(param) and name not in params:
            errors.append(f"missing required parameter: {name}")

    receiver_shape = None
    receiver = case.get("receiver")
    if isinstance(receiver, dict):
        if receiver.get("kind") in {"tensor", "variable"}:
            receiver_shape = list(receiver.get("shape", []))
        elif receiver.get("kind") == "tensor_shape":
            receiver_shape = list(receiver.get("dims", []))
        elif receiver.get("kind") == "tensor_spec":
            receiver_shape = list(receiver.get("shape", []))

    primary_shape = receiver_shape
    for value in params.values():
        if isinstance(value, dict) and value.get("kind") in {"tensor", "variable"}:
            primary_shape = list(value.get("shape", []))
            break

    for name, value in params.items():
        param = spec.params.get(name)
        if not param:
            continue

        low = name.lower()
        if looks_like_dtype_param(name, param) and not isinstance(value, str):
            errors.append(f"{name}: expected dtype string")

        if isinstance(value, dict) and value.get("kind") in {"tensor", "variable"}:
            inferred = infer_data_shape(value.get("data"))
            if inferred is None:
                errors.append(f"{name}: irregular tensor data")
            elif list(inferred) != list(value.get("shape", [])):
                errors.append(
                    f"{name}: declared shape {value.get('shape')} != data shape {inferred}")

        if low == "mask" and isinstance(value, dict) and value.get("kind") == "tensor" and primary_shape:
            if list(value.get("shape", [])) != list(primary_shape):
                errors.append("mask: expected same shape as input tensor")

        if low == "indices" and isinstance(value, dict) and value.get("kind") == "tensor" and receiver_shape:
            if value.get("dtype") not in {"int32", "int64"}:
                errors.append("indices: expected int32/int64 tensor")
            if len(value.get("shape", [])) == 2 and len(receiver_shape) >= 1:
                if value["shape"][1] > len(receiver_shape):
                    errors.append(
                        "indices: last dimension too large for receiver rank")

        if low == "rank" and receiver_shape is not None and isinstance(value, int):
            if value < 0:
                errors.append("rank: must be nonnegative")

        if low == "shape" and isinstance(value, list):
            if any(not isinstance(x, int) or x < 0 for x in value):
                errors.append("shape: expected nonnegative integer dimensions")

    return errors


def resolve_dotted(path: str) -> Any:
    path = normalize_module_path(path)
    parts = path.split(".")
    last_exc: Optional[Exception] = None
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        attrs = parts[i:]
        try:
            obj = importlib.import_module(module_name)
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except Exception as exc:
            last_exc = exc
    raise ImportError(path) from last_exc


def framework_dtype(framework: str, dtype_name: str) -> Any:
    dtype_name = (dtype_name or "float32").lower()
    if framework == "tensorflow":
        import tensorflow as tf
        return getattr(tf, dtype_name)
    if framework == "torch":
        import torch
        return getattr(torch, dtype_name)
    if framework == "jax":
        import jax.numpy as jnp
        return getattr(jnp, dtype_name)
    return dtype_name


def materialize_value(value: Any, framework: str, device: str = "cpu") -> Any:
    if isinstance(value, list):
        return [materialize_value(v, framework, device=device) for v in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("kind")

    if kind in {"tensor", "variable"}:
        data = _decode_complex(value["data"])
        dtype = framework_dtype(framework, value["dtype"])
        if framework == "tensorflow":
            import tensorflow as tf
            dev = "/GPU:0" if device == "gpu" else "/CPU:0"
            with tf.device(dev):
                t = tf.constant(data, dtype=dtype)
                return tf.Variable(t) if kind == "variable" else t
        if framework == "torch":
            import torch
            dev = "cuda" if device == "gpu" and torch.cuda.is_available() else "cpu"
            t = torch.tensor(data, dtype=dtype, device=dev)
            return torch.nn.Parameter(t) if kind == "variable" else t
        if framework == "jax":
            import jax
            import jax.numpy as jnp
            arr = jnp.array(data, dtype=dtype)
            if device == "gpu":
                gpus = [d for d in jax.devices() if d.platform == "gpu"]
                if gpus:
                    return jax.device_put(arr, gpus[0])
            return arr
        return np.asarray(data, dtype=dtype)

    if kind == "ragged_tensor":
        import tensorflow as tf
        dev = "/GPU:0" if device == "gpu" else "/CPU:0"
        with tf.device(dev):
            return tf.ragged.constant(value["data"], dtype=framework_dtype(framework, value["dtype"]))

    if kind == "sparse_tensor":
        import tensorflow as tf
        dev = "/GPU:0" if device == "gpu" else "/CPU:0"
        with tf.device(dev):
            return tf.SparseTensor(
                indices=value["indices"],
                values=tf.constant(value["values"], dtype=framework_dtype(
                    framework, value["dtype"])),
                dense_shape=value["dense_shape"],
            )

    if kind == "tensor_array":
        import tensorflow as tf
        dev = "/GPU:0" if device == "gpu" else "/CPU:0"
        with tf.device(dev):
            ta = tf.TensorArray(dtype=framework_dtype(framework, value["dtype"]), size=int(
                value["size"]), clear_after_read=False)
            for i, item in enumerate(value["elements"]):
                ta = ta.write(i, tf.constant(
                    item, dtype=framework_dtype(framework, value["dtype"])))
            return ta

    if kind == "tensor_shape":
        import tensorflow as tf
        return tf.TensorShape(value["dims"])

    if kind == "tensor_spec":
        import tensorflow as tf
        return tf.TensorSpec(shape=value["shape"], dtype=framework_dtype(framework, value["dtype"]))

    if kind == "dtype_obj":
        return framework_dtype(framework, value["value"])

    return value


def resolve_callable(spec: APISpec, case: Dict[str, Any], device: str = "cpu") -> Tuple[Any, Optional[Any]]:
    framework = case["framework"]
    receiver_obj = materialize_value(
        case["receiver"], framework, device=device) if "receiver" in case else None
    if receiver_obj is not None:
        return getattr(receiver_obj, spec.api_name), receiver_obj
    owner = resolve_dotted(spec.module_path)
    return getattr(owner, spec.api_name), None


def build_call(fn: Any, spec: APISpec, case: Dict[str, Any], device: str = "cpu") -> Tuple[List[Any], Dict[str, Any]]:
    framework = case["framework"]
    materialized = {k: materialize_value(
        v, framework, device=device) for k, v in case.get("parameters", {}).items()}
    try:
        sig = inspect.signature(fn)
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        for name in spec.params:
            if name not in materialized:
                continue
            p = sig.parameters.get(name)
            if p is None or p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                args.append(materialized[name])
            else:
                kwargs[name] = materialized[name]
        return args, kwargs
    except Exception:
        return [materialized[name] for name in spec.params if name in materialized], {}


def summarize_output(obj: Any) -> Dict[str, Any]:
    arr = to_numpy_maybe(obj)
    if arr is not None:
        flat = arr.reshape(-1) if arr.ndim else np.asarray([arr.item()])
        preview = [repr(x) for x in flat[:8].tolist()]
        return {
            "kind": type(obj).__name__,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "preview": preview,
        }
    if isinstance(obj, (list, tuple)):
        return {"kind": type(obj).__name__, "length": len(obj), "preview": [repr(x)[:80] for x in obj[:5]]}
    return {"kind": type(obj).__name__, "repr": repr(obj)[:200]}


def to_numpy_maybe(obj: Any) -> Optional[np.ndarray]:
    try:
        import tensorflow as tf
        if isinstance(obj, tf.Tensor):
            return obj.numpy()
        if isinstance(obj, tf.Variable):
            return obj.numpy()
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import jax
        if isinstance(obj, jax.Array):
            return np.asarray(obj)
    except Exception:
        pass
    try:
        return np.asarray(obj)
    except Exception:
        return None


def contains_nan_or_inf(obj: Any) -> bool:
    arr = to_numpy_maybe(obj)
    if arr is None:
        return False
    try:
        return bool(np.isnan(arr).any() or np.isinf(arr).any()) if arr.dtype.kind in {"f", "c"} else False
    except Exception:
        return False


def values_close(a: Any, b: Any) -> bool:
    arr_a = to_numpy_maybe(a)
    arr_b = to_numpy_maybe(b)
    if arr_a is None or arr_b is None:
        return repr(a) == repr(b)
    if arr_a.shape != arr_b.shape:
        return False
    if arr_a.dtype.kind in {"f", "c"} or arr_b.dtype.kind in {"f", "c"}:
        return bool(np.allclose(arr_a, arr_b, equal_nan=True, rtol=1e-4, atol=1e-5))
    return bool(np.array_equal(arr_a, arr_b))


def can_compare_gpu(framework: str) -> bool:
    try:
        if framework == "tensorflow":
            import tensorflow as tf
            return bool(tf.config.list_physical_devices("GPU"))
        if framework == "torch":
            import torch
            return torch.cuda.is_available()
        if framework == "jax":
            import jax
            return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False
    return False


def execute_case(spec: APISpec, case: Dict[str, Any], compare_gpu: bool = False) -> RunResult:
    start = time.perf_counter()
    try:
        fn, _ = resolve_callable(spec, case, device="cpu")
        args, kwargs = build_call(fn, spec, case, device="cpu")
        out = fn(*args, **kwargs)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        result = RunResult(
            success=True,
            runtime_ms=runtime_ms,
            output_summary=summarize_output(out),
            has_nan_or_inf=contains_nan_or_inf(out),
        )
        if compare_gpu and can_compare_gpu(case["framework"]):
            try:
                fn_gpu, _ = resolve_callable(spec, case, device="gpu")
                args_gpu, kwargs_gpu = build_call(
                    fn_gpu, spec, case, device="gpu")
                out_gpu = fn_gpu(*args_gpu, **kwargs_gpu)
                if not values_close(out, out_gpu):
                    result.inconsistency = True
                    result.inconsistency_detail = "cpu/gpu outputs differ"
            except Exception as exc:
                result.inconsistency = True
                result.inconsistency_detail = f"cpu/gpu compare failed: {type(exc).__name__}: {exc}"
        return result
    except Exception as exc:
        return RunResult(
            success=False,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            output_summary={},
            has_nan_or_inf=False,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
        )


def render_payload(value: Any, framework: str) -> str:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "tensor":
            if framework == "tensorflow":
                return f"tf.constant({repr(value['data'])}, dtype=tf.{value['dtype']})"
            if framework == "torch":
                return f"torch.tensor({repr(value['data'])}, dtype=torch.{value['dtype']})"
            if framework == "jax":
                return f"jnp.array({repr(value['data'])}, dtype=jnp.{value['dtype']})"
            return repr(value["data"])
        if kind == "variable":
            if framework == "tensorflow":
                return f"tf.Variable(tf.constant({repr(value['data'])}, dtype=tf.{value['dtype']}))"
            if framework == "torch":
                return f"torch.nn.Parameter(torch.tensor({repr(value['data'])}, dtype=torch.{value['dtype']}))"
        if kind == "ragged_tensor":
            return f"tf.ragged.constant({repr(value['data'])}, dtype=tf.{value['dtype']})"
        if kind == "sparse_tensor":
            return f"tf.SparseTensor(indices={value['indices']}, values={value['values']}, dense_shape={value['dense_shape']})"
        if kind == "tensor_array":
            return f"TensorArray(dtype={value['dtype']}, size={value['size']})"
        if kind == "tensor_shape":
            return f"tf.TensorShape({value['dims']})"
        if kind == "tensor_spec":
            return f"tf.TensorSpec(shape={value['shape']}, dtype=tf.{value['dtype']})"
        if kind == "dtype_obj":
            if framework == "tensorflow":
                return f"tf.{value['value']}"
            if framework == "torch":
                return f"torch.{value['value']}"
            if framework == "jax":
                return f"jnp.{value['value']}"
    return repr(value)


def render_call(spec: APISpec, case: Dict[str, Any]) -> str:
    framework = case["framework"]
    params = ", ".join(f"{name}={render_payload(value, framework)}" for name,
                       value in case.get("parameters", {}).items())
    module_path = normalize_module_path(spec.module_path)
    pretty_module = module_path.replace(
        "tensorflow", "tf").replace("jax.numpy", "jnp")
    if "receiver" in case:
        receiver_text = render_payload(case["receiver"], framework)
        return f"{receiver_text}.{spec.api_name}({params})"
    return f"{pretty_module}.{spec.api_name}({params})"


def _numpy_dtype(dtype_name: str) -> Any:
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "bfloat16": np.float32,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "string": object,
    }
    return mapping.get(dtype_name, object)


def _clamp_scalar_for_mutation(value: float, bounds: Optional[Dict[str, Any]], likely_valid: bool) -> float:
    if not bounds:
        return value
    if likely_valid:
        return bound_scalar(value, bounds)
    if "max" in bounds and not math.isinf(bounds["max"]):
        return bounds["max"] + 1.0
    if "min" in bounds and not math.isinf(bounds["min"]):
        return bounds["min"] - 1.0
    return value


class Mutator:
    def __init__(self, seed: int = 1234) -> None:
        self.rng = random.Random(seed)

    def mutate(self, spec: APISpec, case: Dict[str, Any], step: int, history: MutationHistory) -> Tuple[Dict[str, Any], MutationRecord]:
        mutant = copy.deepcopy(case)
        names = [name for name in spec.params if name in mutant.get(
            "parameters", {})]
        target = self.rng.choice(names)
        family = ["type", "size", "value"][(step - 1) % 3]
        strategy = self.choose_strategy(family)
        likely_valid = False if history.seen_bad(
            mutant["parameters"][target]) else self.rng.random() < 0.8
        before = copy.deepcopy(mutant["parameters"][target])
        after = self.mutate_value(
            target, spec.params[target], before, family, strategy, likely_valid, case)
        mutant["parameters"][target] = after
        record = MutationRecord(
            step=step,
            parameter=target,
            family=family,
            strategy=strategy,
            likely_valid=likely_valid,
            before=json_safe(before),
            after=json_safe(after),
        )
        return mutant, record

    def choose_strategy(self, family: str) -> str:
        if family == "type":
            return self.rng.choice(["dtype_change", "scalar_cast"])
        if family == "size":
            return self.rng.choice(["grow", "shrink", "rank_up", "rank_down"])
        return self.rng.choice(["noise", "masking", "division"])

    def mutate_value(self, name: str, param: ParamSpec, value: Any, family: str, strategy: str, likely_valid: bool, case: Dict[str, Any]) -> Any:
        bounds = parse_numeric_range(param.description, *param.constraints)
        low = name.lower()

        if isinstance(value, dict) and value.get("kind") in {"tensor", "variable"}:
            return self._mutate_tensor(name, value, param, family, strategy, bounds, likely_valid, case)

        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            out = copy.deepcopy(value)
            idx = self.rng.randrange(len(out))
            out[idx] = self._mutate_tensor(
                name, out[idx], param, family, strategy, bounds, likely_valid, case)
            return out

        if isinstance(value, bool):
            return not value

        # --- Type mutation / size mutation / value mutation for scalar ints ---
        if isinstance(value, int) and not isinstance(value, bool):
            if family == "type":
                return float(value)
            if family == "size":
                return max(0, value * 2 if strategy in {"grow", "rank_up"} else value // 2)
            if strategy == "masking":
                return 0
            if strategy == "division":
                return int(value / max(1, self.rng.choice([2, 4, 8])))
            return int(_clamp_scalar_for_mutation(value + self.rng.choice([-3, -1, 1, 3]), bounds, likely_valid))

        # --- Type mutation / size mutation / value mutation for scalar floats ---
        if isinstance(value, float):
            if family == "type":
                return int(value)
            if family == "size":
                return value * 2 if strategy in {"grow", "rank_up"} else value / 2
            if strategy == "masking":
                return 0.0
            if strategy == "division":
                return value / self.rng.choice([2.0, 4.0, 8.0])
            mutated = value + self.rng.uniform(-1.0, 1.0)
            return _clamp_scalar_for_mutation(mutated, bounds, likely_valid)

        if isinstance(value, str):
            if family == "type":
                candidates = allowed_dtypes(
                    param.type, param.description, *param.constraints)
                if candidates:
                    choices = [x for x in candidates if x !=
                               value] or candidates
                    return self.rng.choice(choices)
            return value + "_m"

        if isinstance(value, list):
            out = copy.deepcopy(value)
            if family == "size":
                if strategy in {"grow", "rank_up"}:
                    out.append(copy.deepcopy(out[-1]) if out else 1)
                elif out:
                    out = out[:-1] or out
                return out
            if out:
                idx = self.rng.randrange(len(out))
                if isinstance(out[idx], (int, float)):
                    out[idx] = out[idx] + 1
            return out

        return value

    def _mutate_tensor(
        self,
        name: str,
        value: Dict[str, Any],
        param: ParamSpec,
        family: str,
        strategy: str,
        bounds: Optional[Dict[str, Any]],
        likely_valid: bool,
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = copy.deepcopy(value)
        dtype_name = out["dtype"]
        arr = np.asarray(_decode_complex(
            out["data"]), dtype=_numpy_dtype(dtype_name))
        low = name.lower()

        # --- Type mutation for tensors ---
        if family == "type":
            candidates = allowed_dtypes(
                param.type, param.description, *param.constraints)
            if low == "mask":
                candidates = ["bool"] if likely_valid else ["bool", "float32"]
            elif low in {"indices", "row_splits", "row_lengths", "value_rowids", "rowids", "nested_row_splits", "nested_row_lengths", "nested_value_rowids"}:
                candidates = [x for x in candidates if x in INT_DTYPES] or [
                    "int32", "int64"]
            elif not candidates:
                candidates = [dtype_name, "float32", "int32", "bool"]
            choices = [x for x in candidates if x !=
                       dtype_name] or [dtype_name]
            new_dtype = self.rng.choice(choices)
            out["dtype"] = new_dtype
            if new_dtype == "bool":
                arr = np.asarray(arr, dtype=np.float64) > 0
            elif new_dtype in INT_DTYPES:
                arr = np.rint(np.asarray(arr, dtype=np.float64)
                              ).astype(_numpy_dtype(new_dtype))
            elif new_dtype == "string":
                arr = np.full(arr.shape, "x", dtype=object)
            else:
                arr = np.asarray(arr).astype(_numpy_dtype(new_dtype))

        # --- Size mutation for tensors ---
        elif family == "size":
            if low == "mask" and likely_valid:
                target_shape = None
                receiver = case.get("receiver")
                if isinstance(receiver, dict) and receiver.get("kind") in {"tensor", "variable"}:
                    target_shape = list(receiver.get("shape", []))
                if target_shape is None:
                    for other_name, other_value in case.get("parameters", {}).items():
                        if other_name != name and isinstance(other_value, dict) and other_value.get("kind") in {"tensor", "variable"}:
                            target_shape = list(other_value.get("shape", []))
                            break
                if target_shape:
                    arr = np.resize(arr, target_shape)
                else:
                    arr = np.resize(arr, [2, 2])
            else:
                shape = list(arr.shape)
                if not shape:
                    shape = [1]
                if strategy == "grow":
                    shape = [max(1, d * 2) for d in shape]
                elif strategy == "shrink":
                    shape = [max(1, d // 2) for d in shape]
                elif strategy == "rank_up":
                    shape = [1] + shape
                elif strategy == "rank_down":
                    shape = shape[1:] if len(shape) > 1 else [1]
                arr = np.resize(arr, shape)

        # --- Value mutation for tensors: noise / masking / division ---
        else:
            if strategy == "noise":
                arr = np.asarray(arr, dtype=np.float64) + np.random.default_rng(
                    self.rng.randint(0, 2**31 - 1)).normal(0.0, 0.3, size=np.asarray(arr).shape)
            elif strategy == "masking":
                arr = np.asarray(arr).copy()
                flat = arr.reshape(-1)
                if flat.size:
                    picks = np.random.default_rng(self.rng.randint(
                        0, 2**31 - 1)).choice(flat.size, max(1, flat.size // 3), replace=False)
                    for idx in picks:
                        flat[idx] = 0
            else:
                arr = np.asarray(arr, dtype=np.float64) / \
                    self.rng.choice([2, 4, 8])

            if bounds:
                if likely_valid:
                    if "min" in bounds and not math.isinf(bounds["min"]):
                        min_val = bounds["min"] if bounds.get(
                            "min_inclusive", True) else bounds["min"] + 1e-3
                        arr = np.maximum(arr, min_val)
                    if "max" in bounds and not math.isinf(bounds["max"]):
                        max_val = bounds["max"] if bounds.get(
                            "max_inclusive", True) else bounds["max"] - 1e-3
                        arr = np.minimum(arr, max_val)
                else:
                    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
                    if flat.size:
                        if "max" in bounds and not math.isinf(bounds["max"]):
                            flat[0] = bounds["max"] + 1.0
                        elif "min" in bounds and not math.isinf(bounds["min"]):
                            flat[0] = bounds["min"] - 1.0
                        arr = flat.reshape(np.asarray(arr).shape)

        if out["dtype"] in INT_DTYPES:
            arr = np.rint(np.asarray(arr, dtype=np.float64)
                          ).astype(_numpy_dtype(out["dtype"]))
        elif out["dtype"] == "bool":
            arr = np.asarray(arr).astype(np.bool_)

        out["shape"] = list(np.asarray(arr).shape)
        out["data"] = _encode_complex(arr) if out["dtype"] in COMPLEX_DTYPES or np.asarray(
            arr).dtype.kind == "c" else np.asarray(arr).tolist()
        return out


class CoverageTracker:
    def __init__(self, output_dir: Path, source_packages: Sequence[str]) -> None:
        self.output_dir = output_dir
        self.source_packages = [
            x for x in source_packages if x and x != "python"]
        self.cov = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import coverage
            self.cov = coverage.Coverage(
                data_file=str(self.output_dir / ".coverage"),
                branch=True,
                config_file=False,
                source=self.source_packages or None,
            )
            self.cov.start()
        except Exception:
            self.cov = None

    def warmup_imports(self, frameworks: Sequence[str]) -> None:
        # Import frameworks outside per-API contexts so import/startup coverage does not get charged
        # to the first API that runs.
        for framework in frameworks:
            try:
                if framework == "tensorflow":
                    import tensorflow  # noqa: F401
                elif framework == "torch":
                    import torch  # noqa: F401
                elif framework == "jax":
                    import jax  # noqa: F401
                    import jax.numpy  # noqa: F401
            except Exception:
                pass

    def switch(self, context: str) -> None:
        if self.cov is not None:
            self.cov.switch_context(context)

    def stop(self) -> None:
        if self.cov is not None:
            self.cov.stop()
            self.cov.save()

    def summarize(self, apis: Sequence[str]) -> Dict[str, Any]:
        if self.cov is None:
            return {"enabled": False}

        data = self.cov.get_data()
        files = sorted(self._filtered_files(data.measured_files()))
        executable_by_file: Dict[str, int] = {}
        for filename in files:
            try:
                _, executable, _, _, _ = self.cov.analysis2(filename)
                executable_by_file[filename] = len(executable)
            except Exception:
                executable_by_file[filename] = 0

        def ctx_summary(pattern: str) -> Dict[str, Any]:
            data.set_query_contexts([pattern])
            rows: List[Dict[str, Any]] = []
            covered_total = 0
            executable_total = 0
            for filename in files:
                executable = executable_by_file.get(filename, 0)
                if executable <= 0:
                    continue
                executed = data.lines(filename) or []
                if not executed:
                    continue
                covered = min(len(set(executed)), executable)
                covered_total += covered
                executable_total += executable
                rows.append({
                    "file": filename,
                    "covered_lines": covered,
                    "executable_lines": executable,
                    "coverage_percent": round(100.0 * covered / executable, 2),
                })
            rows.sort(key=lambda x: (-x["coverage_percent"], x["file"]))
            return {
                "coverage_percent": round(100.0 * covered_total / executable_total, 4) if executable_total else 0.0,
                "covered_lines": covered_total,
                "executable_lines": executable_total,
                "files_touched": len(rows),
                "top_files": rows[:20],
            }

        summary = {
            "enabled": True,
            "source_packages": self.source_packages,
            "overall": ctx_summary(r"^api::"),
            "per_api": {},
        }
        for api in apis:
            summary["per_api"][api] = ctx_summary(rf"^api::{re.escape(api)}::")
        return summary

    def _filtered_files(self, files: Sequence[str]) -> List[str]:
        if not self.source_packages:
            return list(files)
        selected: List[str] = []
        for filename in files:
            normalized = filename.replace("\\", "/")
            for pkg in self.source_packages:
                token = "/" + pkg.replace(".", "/") + "/"
                token_file = "/" + pkg.replace(".", "/") + ".py"
                if token in normalized or normalized.endswith(token_file):
                    selected.append(filename)
                    break
        return selected


def classify_issue(result: RunResult, valid: bool) -> str:
    if not valid:
        return "validation_error"
    if not result.success:
        return "exception"
    if result.inconsistency:
        return "inconsistency"
    if result.has_nan_or_inf:
        return "nan_or_inf"
    return "success"


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    return repr(obj)


def bug_candidate(issue: str, mutation: Optional[MutationRecord], baseline: bool = False) -> bool:
    if issue in {"nan_or_inf", "inconsistency"}:
        return True
    if baseline:
        return False
    if issue == "exception" and mutation is not None and mutation.likely_valid:
        return True
    return False


class Stage4Runner:
    def __init__(
        self,
        spec_dir: str,
        results_dir: str,
        api_filter: str = "",
        max_apis: int = 0,
        iterations: int = 20,
        seed: int = 1234,
        include_optional: bool = False,
        compare_gpu: bool = False,
    ) -> None:
        self.spec_dir = spec_dir
        self.results_dir = Path(results_dir)
        self.api_filter = api_filter
        self.max_apis = max_apis
        self.iterations = iterations
        self.include_optional = include_optional
        self.compare_gpu = compare_gpu
        self.rng_seed = seed
        self.specs = self._select_specs()
        packages = sorted({detect_framework(spec)
                          for spec in self.specs.values()})
        self.coverage = CoverageTracker(
            self.results_dir / "coverage", packages)
        self.mutator = Mutator(seed=seed)

    def _select_specs(self) -> Dict[str, APISpec]:
        specs = load_api_specs(self.spec_dir)
        items = list(specs.items())
        if self.api_filter:
            items = [(k, v) for k, v in items if k ==
                     self.api_filter or v.api_name == self.api_filter]
        if self.max_apis > 0:
            items = items[: self.max_apis]
        return dict(items)

    def run(self) -> Dict[str, Any]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        per_api_dir = self.results_dir / "per_api"
        per_api_dir.mkdir(parents=True, exist_ok=True)
        bugs_path = self.results_dir / "bug_candidates.jsonl"

        overall = Counter()
        failure_reasons = Counter()
        bug_rows: List[Dict[str, Any]] = []
        api_rollup: Dict[str, Dict[str, Any]] = {}

        self.coverage.start()
        self.coverage.warmup_imports(
            sorted({detect_framework(spec) for spec in self.specs.values()}))

        for api, spec in self.specs.items():
            framework = detect_framework(spec)
            seed_case = generate_seed_case(
                spec, include_optional=self.include_optional)
            seed_errors = validate_case(spec, seed_case)
            api_rows: List[Dict[str, Any]] = []

            baseline_context = f"api::{api}::baseline"
            if seed_errors:
                baseline = RunResult(False, 0.0, {}, False,
                                     "ValidationError", "; ".join(seed_errors))
                baseline_issue = classify_issue(baseline, False)
            else:
                self.coverage.switch(baseline_context)
                baseline = execute_case(
                    spec, seed_case, compare_gpu=self.compare_gpu)
                baseline_issue = classify_issue(baseline, True)

            baseline_row = {
                "kind": "baseline",
                "api": api,
                "framework": framework,
                "valid": not seed_errors,
                "issue": baseline_issue,
                "call": render_call(spec, seed_case),
                "case": json_safe(seed_case),
                "result": json_safe(asdict(baseline)),
                "validation_errors": seed_errors,
            }
            api_rows.append(baseline_row)

            overall["apis_total"] += 1
            overall["baseline_attempts"] += 1
            overall[f"baseline_{baseline_issue}"] += 1
            if baseline_issue != "success":
                failure_reasons[f"baseline::{baseline.exception_type or baseline_issue}"] += 1

            mutants_attempted = 0
            mutants_success = 0
            current_case = copy.deepcopy(seed_case)
            runner_seed_problem = bool(seed_errors) or (baseline_issue == "exception" and baseline.exception_type in {
                "TypeError", "AttributeError", "ImportError", "ValidationError"})

            if baseline.success and not seed_errors:
                history = MutationHistory()
                for step in range(1, self.iterations + 1):
                    mutant_case, mutation = self.mutator.mutate(
                        spec, current_case, step, history)
                    errors = validate_case(spec, mutant_case)
                    if errors:
                        result = RunResult(
                            False, 0.0, {}, False, "ValidationError", "; ".join(errors))
                        issue = classify_issue(result, False)
                    else:
                        ctx = f"api::{api}::mutant::{step}"
                        self.coverage.switch(ctx)
                        result = execute_case(
                            spec, mutant_case, compare_gpu=self.compare_gpu)
                        issue = classify_issue(result, True)

                    row = {
                        "kind": "mutant",
                        "step": step,
                        "api": api,
                        "framework": framework,
                        "valid": not errors,
                        "issue": issue,
                        "call": render_call(spec, mutant_case),
                        "mutation": json_safe(asdict(mutation)),
                        "case": json_safe(mutant_case),
                        "result": json_safe(asdict(result)),
                        "validation_errors": errors,
                    }
                    api_rows.append(row)

                    overall["mutant_attempts"] += 1
                    overall[f"mutant_{issue}"] += 1
                    mutants_attempted += 1
                    if result.success:
                        mutants_success += 1
                        current_case = mutant_case
                    else:
                        failure_reasons[f"mutant::{result.exception_type or issue}"] += 1

                    history.record(mutation.family, result.success,
                                   mutant_case["parameters"][mutation.parameter])

                    if bug_candidate(issue, mutation):
                        bug = {
                            "api": api,
                            "framework": framework,
                            "step": step,
                            "issue": issue,
                            "exception_type": result.exception_type,
                            "exception_message": result.exception_message,
                            "mutation": json_safe(asdict(mutation)),
                            "call": render_call(spec, mutant_case),
                            "source_file": spec.source_file,
                        }
                        bug_rows.append(bug)

            api_rollup[api] = {
                "framework": framework,
                "baseline_issue": baseline_issue,
                "baseline_success": baseline.success and not seed_errors,
                "mutants_attempted": mutants_attempted,
                "mutants_success": mutants_success,
                "runner_seed_problem": runner_seed_problem,
            }
            (per_api_dir / f"{sanitize_filename(api)}.json").write_text(
                json.dumps(api_rows, indent=2, ensure_ascii=False), encoding="utf-8")

        self.coverage.stop()

        if bug_rows:
            with bugs_path.open("w", encoding="utf-8") as fh:
                for row in bug_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        coverage_summary = self.coverage.summarize(list(self.specs))

        suspected_seed_failures = sum(
            1 for row in api_rollup.values() if row["runner_seed_problem"])
        successful_baselines = sum(
            1 for row in api_rollup.values() if row["baseline_success"])
        successful_mutant_apis = sum(
            1 for row in api_rollup.values() if row["mutants_success"] > 0)

        summary = {
            "overall": {
                **overall,
                "successful_baseline_apis": successful_baselines,
                "successful_mutant_apis": successful_mutant_apis,
                "suspected_runner_seed_failures": suspected_seed_failures,
                "failure_reasons": failure_reasons.most_common(50),
                "bug_candidates": len(bug_rows),
            },
            "coverage_overview": {
                "percent": coverage_summary.get("overall", {}).get("coverage_percent", 0.0),
                "covered_lines": coverage_summary.get("overall", {}).get("covered_lines", 0),
                "executable_lines": coverage_summary.get("overall", {}).get("executable_lines", 0),
                "files_touched": coverage_summary.get("overall", {}).get("files_touched", 0),
            },
            "api_rollup": api_rollup,
            "coverage": coverage_summary,
            "generated_files": {
                "per_api_dir": str(per_api_dir),
                "bug_candidates_jsonl": str(bugs_path),
                "coverage_dir": str(self.results_dir / "coverage"),
            },
            "notes": [
                "This runner keeps Stage 4 simple: JSON spec -> in-memory seed -> execute -> mutate -> execute.",
                "It saves readable per-API traces and bug candidates, but does not create separate .init.json files.",
                "Coverage here is Python-layer coverage via coverage.py. Native/C++ coverage still needs an instrumented source build.",
                "Coverage is reported as numeric totals plus top touched files, and import warmup is kept outside per-API contexts.",
            ],
        }
        (self.results_dir / "summary.json").write_text(json.dumps(json_safe(summary),
                                                                  indent=2, ensure_ascii=False), encoding="utf-8")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified Stage 4 runner")
    parser.add_argument("--spec-dir", required=True,
                        help="Directory with Stage 2/3 JSON specs")
    parser.add_argument("--results-dir", required=True,
                        help="Directory for reports")
    parser.add_argument("--api", default="", help="Optional single API filter")
    parser.add_argument("--max-apis", type=int, default=0,
                        help="Optional cap on number of APIs")
    parser.add_argument("--iterations", type=int,
                        default=20, help="Mutations per API")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--include-optional", action="store_true",
                        help="Include optional params without defaults")
    parser.add_argument("--compare-gpu", action="store_true",
                        help="Compare CPU and GPU outputs when GPU exists")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = Stage4Runner(
        spec_dir=args.spec_dir,
        results_dir=args.results_dir,
        api_filter=args.api,
        max_apis=args.max_apis,
        iterations=args.iterations,
        seed=args.seed,
        include_optional=args.include_optional,
        compare_gpu=args.compare_gpu,
    )
    summary = runner.run()
    print(json.dumps({
        "overall": summary["overall"],
        "coverage_overview": summary["coverage_overview"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
