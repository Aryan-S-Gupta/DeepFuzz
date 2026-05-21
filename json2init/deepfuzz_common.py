from __future__ import annotations

import base64
import inspect
import importlib
import json
import math
import os
import random
import re
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from json2init.adapters import get_adapter
except Exception:  # pragma: no cover - script/local import fallback
    from adapters import get_adapter  # type: ignore

try:
    from common.result_io import atomic_write_json
except Exception:  # pragma: no cover
    atomic_write_json = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

SIMPLE_TYPE_PRIORITY = [
    "bool", "int", "float", "number", "string", "str", "tensor", "list", "tuple",
    "sequence", "dict", "dtype", "device", "bytes", "any",
]

UNSUPPORTED_TOKENS = {
    "future", "callable", "executor", "dataset", "dataloader", "module", "optimizer",
    "model", "hook", "callback",
}

ENVIRONMENT_ERROR_PATTERNS = [
    "torch not compiled with cuda enabled",
    "backend doesn't support",
    "backend does not support",
    "allocator for mps is not a deviceallocator",
    "nvidia-ml-py does not seem to be installed",
    "not compiled with cuda",
    "cuda driver",
    "cuda runtime",
    "not available on this system",
    "operation not supported on this device",
    "mlir translation rule",
    "not found for platform cpu",
]

# 1x1 PNG bytes, valid for TensorFlow image decoders.
_VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)

# 1x1 baseline JPEG.  Kept as bytes so image decode seeds do not depend on PIL
# or external files.
_VALID_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQ"
    "AAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
    "9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/"
    "ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEA"
    "AgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//E"
    "ABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)

FAKE_PARAM_NAMES = {
    "caution", "note", "warning", "examples", "example", "returns", "return",
    "grads", "types", "matrix", "scatter", "updated", "optional", "required",
}

@dataclass
class ParameterRuntime:
    name: str
    spec: Dict[str, Any]
    chosen_type: str
    include_in_base_call: bool
    base_seed_spec: Dict[str, Any]
    mutation_rules: List[str]
    valid_mutation_rules: List[str] = field(default_factory=list)
    negative_mutation_rules: List[str] = field(default_factory=list)
    unresolved_reason: str = ""
    call_kind: str = "keyword"


@dataclass
class RuntimeAPIObject:
    api_full_name: str
    api_name: str
    module_path: str
    import_path: str
    backend: str
    params: Dict[str, ParameterRuntime]
    output: Dict[str, Any]
    constraints: List[str]
    spec_ready: bool
    ready_for_stage4: bool
    readiness_reasons: List[str] = field(default_factory=list)
    smoke_test: Dict[str, Any] = field(default_factory=dict)
    signature_binding: Dict[str, Any] = field(default_factory=dict)
    param_dependencies: List[str] = field(default_factory=list)
    raise_contract: List[Dict[str, Any]] = field(default_factory=list)
    doc_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_full_name": self.api_full_name,
            "api_name": self.api_name,
            "module_path": self.module_path,
            "import_path": self.import_path,
            "backend": self.backend,
            "params": {k: asdict(v) for k, v in self.params.items()},
            "output": self.output,
            "constraints": self.constraints,
            "spec_ready": self.spec_ready,
            "ready_for_stage4": self.ready_for_stage4,
            "readiness_reasons": self.readiness_reasons,
            "smoke_test": self.smoke_test,
            "signature_binding": self.signature_binding,
            "param_dependencies": self.param_dependencies,
            "raise_contract": self.raise_contract,
            "doc_confidence": self.doc_confidence,
        }


def clean_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", " ", v).strip()


def _json_desanitize(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__kind__") == "bytes_b64" and "data" in value:
            try:
                return base64.b64decode(value["data"])
            except Exception:
                return b""
        return {k: _json_desanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_desanitize(v) for v in value]
    return value


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__kind__": "bytes_b64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = _json_desanitize(data)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} is not an object")
    return data


def write_json(path: str, data: Any) -> None:
    payload = _json_sanitize(data)
    if atomic_write_json is not None:
        atomic_write_json(path, payload)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_json_specs(spec_dir: str) -> Iterable[str]:
    for path in sorted(Path(spec_dir).glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        yield str(path)


def parse_type_tokens(type_str: str) -> List[str]:
    s = clean_text(type_str)
    if not s:
        return []
    raw = re.split(r"[|/,]", s)
    tokens: List[str] = []
    for part in raw:
        p = clean_text(part).lower()
        if not p:
            continue
        p = p.replace("torch.", "").replace("tensorflow.", "tf.").replace("numpy.", "")
        if p in {"str", "string"}:
            norm = "string"
        elif p in {"bytes", "byte", "bytearray"}:
            norm = "bytes"
        elif p in {"int", "integer", "long", "short"}:
            norm = "int"
        elif p in {"float", "double", "real"}:
            norm = "float"
        elif p in {"number", "numeric", "scalar"}:
            norm = "number"
        elif p in {
            "tensor", "bytetensor", "ndarray", "array", "arraylike", "array_like",
            "array-like", "array_like,", "arraylike,", "sparsetensor", "raggedtensor",
            "tensorlike", "jax.array",
        }:
            norm = "tensor"
        elif p == "tuple":
            norm = "tuple"
        elif p == "list":
            norm = "list"
        elif p in {"sequence", "iterable"}:
            norm = "sequence"
        elif p in {"dict", "mapping", "ordereddict"}:
            norm = "dict"
        elif p in {"bool", "boolean"}:
            norm = "bool"
        elif p in {"dtype", "datatype", "tf.dtypes.dtype"}:
            norm = "dtype"
        elif p in {"device", "torch.device"}:
            norm = "device"
        elif p in {"future", "futures"}:
            norm = "future"
        elif p in {"callable", "function", "fn"}:
            norm = "callable"
        elif p in {"any", "object"}:
            norm = "any"
        else:
            norm = p
        if norm not in tokens:
            tokens.append(norm)
    return tokens


def detect_backend(import_path: str, module_path: str = "") -> str:
    p = import_path or module_path
    if p.startswith("tensorflow") or p.startswith("tf"):
        return "tensorflow"
    if p.startswith("torch"):
        return "torch"
    if p.startswith("jax"):
        return "jax"
    return "python"


def is_fake_param_name(name: str) -> bool:
    n = clean_text(name).strip("*` ").lower()
    return not n or n in {"self", "cls"} or n in FAKE_PARAM_NAMES or len(n) > 80


def needs_explicit_self(import_path: str) -> bool:
    return import_path.startswith("jax.numpy.ufunc.") or import_path.startswith("jax.numpy.dtype.")


@dataclass
class RuntimeSignature:
    found: bool
    params: Dict[str, inspect.Parameter]
    accepts_kwargs: bool


def runtime_signature(import_path: str) -> RuntimeSignature:
    if not import_path:
        return RuntimeSignature(False, {}, False)
    try:
        obj = import_api(import_path)
        sig = inspect.signature(obj)
    except Exception:
        return RuntimeSignature(False, {}, False)
    params: Dict[str, inspect.Parameter] = {}
    accepts_kwargs = False
    include_self = needs_explicit_self(import_path)
    for name, param in sig.parameters.items():
        if name in {"self", "cls"} and not include_self:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if not is_fake_param_name(name) or (include_self and name == "self"):
            params[name] = param
    return RuntimeSignature(True, params, accepts_kwargs)


def signature_call_kind(param: Optional[inspect.Parameter]) -> str:
    if param is None:
        return "keyword"
    if param.kind == inspect.Parameter.POSITIONAL_ONLY:
        return "positional_only"
    if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
        return "positional_or_keyword"
    if param.kind == inspect.Parameter.KEYWORD_ONLY:
        return "keyword_only"
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
        return "var_positional"
    if param.kind == inspect.Parameter.VAR_KEYWORD:
        return "var_keyword"
    return "keyword"


def synthesize_spec_for_signature_param(name: str, param: inspect.Parameter, backend: str) -> Dict[str, Any]:
    lname = clean_text(name).lower()
    default = "" if param.default is inspect._empty else clean_text(param.default)
    flag = "Required" if param.default is inspect._empty else "Optional"
    typ = ""
    if lname in {"dtype", "out_type", "output_type"} or lname.endswith("_dtype"):
        typ = "dtype"
    elif lname in {"axis", "dim", "depth", "k", "num", "count", "window_length", "fft_length", "sequence_length", "sequence_stride", "num_bits", "block_size", "kth"}:
        typ = "int"
    elif lname in {"keepdims", "exclusive", "reverse", "canonicalized_coordinates", "narrow_range", "sorted"}:
        typ = "bool"
    elif lname in {"side", "direction", "padding", "data_format", "pattern", "rewrite"}:
        typ = "string"
    elif "shape" in lname or lname in {"size", "dims", "paddings", "crop_size", "strides", "ksize", "kernel_size", "pool_size"}:
        typ = "list"
    elif lname in {"contents", "serialized"}:
        typ = "bytes"
    else:
        typ = "tensor" if backend == "tensorflow" else "any"
    return {
        "type": typ,
        "size": "",
        "default": default,
        "flag": flag,
        "description": f"signature parameter {name}",
        "constraints": [],
    }


def choose_seed_type(type_str: str, description: str = "", param_name: str = "") -> str:
    tokens = parse_type_tokens(type_str)
    if not tokens:
        tokens = parse_type_tokens(description)
    pname = clean_text(param_name).lower()
    dlow = clean_text(description).lower()
    if pname == "name":
        return "string"
    if pname in {"keepdims", "exclusive", "reverse", "canonicalized_coordinates", "narrow_range", "sorted", "use_quote_delim"}:
        return "bool"
    if pname in {"side", "direction", "padding", "data_format", "pattern", "rewrite", "input_encoding", "output_encoding"}:
        return "string"
    if pname in {"alpha", "beta", "gamma", "epsilon", "threshold"}:
        return "float"
    if pname in {"depth", "k", "num", "count", "window_length", "sequence_length", "sequence_stride", "fft_length", "num_bits", "num_buckets", "block_size"}:
        return "int"
    if "sparse" in pname or "ragged" in pname:
        return "tensor"
    if dlow.startswith("when true") or dlow.startswith("whether ") or dlow.startswith("if true"):
        return "bool"
    if pname in {"dtype", "out_type", "output_type"} or pname.endswith("_dtype") or pname.endswith("_type"):
        return "dtype"
    if pname in {"output_types"}:
        return "list"
    if pname in {"serialized", "contents", "bytes"} or "encoded image" in dlow or "serialized tensor" in dlow:
        return "bytes"
    if pname in {"field_names"}:
        return "list"
    if pname in {"context_features", "sequence_features"}:
        return "dict"
    if pname in {"axis", "axes", "dim", "dims"}:
        return "int" if pname in {"axis", "dim"} else "tuple"
    if pname in {"shape", "size", "output_size", "input_size"} or pname.endswith("_shape") or pname.endswith("_size"):
        return "tuple" if "tuple" in dlow else "list"
    if any(tok in pname for tok in ["kernel", "stride", "dilation", "padding"]):
        return "tuple" if "tuple" in dlow else "int"
    for bad in UNSUPPORTED_TOKENS:
        if bad in tokens:
            return bad
    if not tokens:
        return "any"
    for pref in SIMPLE_TYPE_PRIORITY:
        if pref in tokens:
            return pref
    return tokens[0]


def infer_shape_from_size_and_constraints(size: str, constraints: Sequence[str]) -> List[int]:
    text = " ".join([clean_text(size)] + [clean_text(x) for x in constraints]).lower()
    m = re.search(r"\b([1-8])\s*[- ]?d\b", text)
    if m:
        return [1] * int(m.group(1))
    m = re.search(r"shape\s*=\s*\(([^)]*)\)", text)
    if m:
        rank = len([x for x in m.group(1).split(",") if x.strip()])
        return [1] * max(rank, 1)
    if "n-dimensional" in text or "n dimensional" in text:
        return [1]
    if "matrix" in text or "matrices" in text:
        return [2, 2]
    if "image" in text or "height" in text or "width" in text:
        return [2, 2, 2, 3]
    return [1]


def infer_numeric_base(spec: Dict[str, Any], as_float: bool) -> Any:
    text = " ".join(
        [clean_text(spec.get("description", "")), clean_text(spec.get("size", ""))]
        + [clean_text(x) for x in spec.get("constraints", [])]
    ).lower()
    if "between 0 and 1" in text or "values between 0 and 1" in text or "in [0, 1]" in text:
        return 0.5 if as_float else 1
    if "non-negative" in text or ">= 0" in text:
        return 0.0 if as_float else 0
    if "positive" in text or "> 0" in text:
        return 1.0 if as_float else 1
    return 1.0 if as_float else 1


def infer_enum_values(spec: Dict[str, Any]) -> List[str]:
    for key in ("enum_values", "valid_values", "enum"):
        vals = spec.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        if isinstance(vals, list):
            clean = [clean_text(x) for x in vals if clean_text(x)]
            if clean:
                return list(dict.fromkeys(clean))
    desc = clean_text(spec.get("description", ""))
    found = re.findall(r"[`'\"]([A-Za-z0-9_.-]+)[`'\"]", desc)
    if len(found) >= 2:
        return list(dict.fromkeys(found))
    return []


def jax_padding_enum_values(api_full_name: str, param_name: str) -> List[str]:
    if api_full_name.startswith("jax.") and clean_text(param_name).lower() == "padding":
        return ["VALID", "SAME", "SAME_LOWER"]
    return []


def build_base_seed_spec(api_full_name: str, param_name: str, spec: Dict[str, Any], backend: str) -> Tuple[Dict[str, Any], str, str]:
    chosen = choose_seed_type(spec.get("type", ""), spec.get("description", ""), param_name)
    pname = clean_text(param_name).lower()
    desc = clean_text(spec.get("description", "")).lower()
    constraint_text = " ".join(clean_text(x) for x in spec.get("constraints", [])).lower()

    recipe = get_adapter(backend).recipe(api_full_name, param_name, spec, chosen, backend)
    if recipe is not None:
        return recipe, chosen, ""

    if chosen in UNSUPPORTED_TOKENS:
        return {}, chosen, f"unsupported base seed type: {chosen}"
    if chosen == "bool":
        return {"kind": "literal", "value": False}, chosen, ""
    if chosen == "int":
        base = int(infer_numeric_base(spec, as_float=False))
        if pname in {"k", "depth", "num", "count", "window_length", "sequence_length", "sequence_stride", "fft_length", "num_bits", "num_buckets", "block_size"}:
            base = 2 if pname not in {"k", "sequence_stride", "num_bits"} else (8 if pname == "num_bits" else 1)
        return {"kind": "literal", "value": max(1, base)}, chosen, ""
    if chosen in {"float", "number"}:
        if pname == "epsilon":
            return {"kind": "literal", "value": 1e-7}, chosen, ""
        if pname in {"alpha", "beta", "gamma", "threshold"}:
            return {"kind": "literal", "value": 0.5}, chosen, ""
        return {"kind": "literal", "value": infer_numeric_base(spec, as_float=True)}, chosen, ""
    if chosen == "string":
        enums = jax_padding_enum_values(api_full_name, param_name) or infer_enum_values(spec)
        if pname == "side":
            return {"kind": "literal", "value": "left"}, chosen, ""
        if pname == "direction":
            return {"kind": "literal", "value": "ASCENDING" if api_full_name.startswith("tensorflow.") else "ascending"}, chosen, ""
        if pname == "padding":
            return {"kind": "literal", "value": "VALID"}, chosen, ""
        if pname == "data_format":
            return {"kind": "literal", "value": "NHWC" if api_full_name.startswith("tensorflow.") else "channels_last"}, chosen, ""
        if pname == "pattern":
            return {"kind": "literal", "value": "a.*"}, chosen, ""
        if pname == "rewrite":
            return {"kind": "literal", "value": "x"}, chosen, ""
        return {"kind": "literal", "value": enums[0] if enums else "test"}, chosen, ""
    if chosen == "bytes":
        return {"kind": "bytes", "value": _VALID_PNG_BYTES}, chosen, ""
    if chosen == "dtype":
        dtypes = spec.get("dtype_candidates") or []
        value = clean_text(dtypes[0]) if dtypes else "float32"
        return {"kind": "dtype", "value": value}, chosen, ""
    if chosen == "device":
        return {"kind": "device", "value": "cpu"}, chosen, ""
    if chosen == "dict":
        return {"kind": "literal", "value": {}}, chosen, ""
    if chosen == "list":
        if pname in {"shape", "dims", "size", "output_shape", "input_shape"} or pname.endswith("_shape"):
            return {"kind": "literal", "value": [2, 2]}, chosen, ""
        if pname in {"paddings"}:
            return {"kind": "literal", "value": [[0, 0], [0, 0]]}, chosen, ""
        return {"kind": "literal", "value": [1]}, chosen, ""
    if chosen == "tuple":
        if param_name == "pad" or "padding" in desc or "m must be even" in constraint_text or "m is even" in desc:
            return {"kind": "literal", "value": [0, 0], "as_tuple": True}, chosen, ""
        return {"kind": "literal", "value": [1], "as_tuple": True}, chosen, ""
    if chosen == "sequence":
        return {"kind": "literal", "value": [1]}, chosen, ""
    if chosen == "tensor":
        shape = infer_shape_from_size_and_constraints(spec.get("size", ""), spec.get("constraints", []))
        dtypes = spec.get("dtype_candidates") or []
        dtype = clean_text(dtypes[0]) if dtypes else ("uint8" if "bytetensor" in clean_text(spec.get("type", "")).lower() else "float32")
        base_value = infer_numeric_base(spec, as_float=("float" in dtype or "bfloat" in dtype))
        return {"kind": "tensor", "shape": shape, "dtype": dtype, "fill": base_value, "device": "cpu", "backend": backend}, chosen, ""
    if chosen == "any":
        return {"kind": "literal", "value": 1}, chosen, ""
    return {}, chosen, f"unhandled chosen type: {chosen}"


def build_mutation_rules(spec: Dict[str, Any], chosen_type: str) -> List[str]:
    return build_valid_mutation_rules(spec, chosen_type) + build_negative_mutation_rules(spec, chosen_type)


def build_valid_mutation_rules(spec: Dict[str, Any], chosen_type: str) -> List[str]:
    rules: List[str] = []
    enums = infer_enum_values(spec)
    if enums:
        return ["enum_switch"]
    if chosen_type in {"int", "float", "number", "bool"}:
        rules.extend(["scalar_delta"])
    elif chosen_type == "string":
        rules.extend(["string_replace"])
    elif chosen_type in {"list", "tuple", "sequence"}:
        rules.extend(["structure_grow", "element_delta"])
    elif chosen_type == "tensor":
        rules.extend(["value_noise", "value_mask", "value_division"])
    elif chosen_type in {"dict", "dtype", "device", "any", "bytes"}:
        rules.append("noop_clone")
    return list(dict.fromkeys(rules))


def build_negative_mutation_rules(spec: Dict[str, Any], chosen_type: str) -> List[str]:
    rules: List[str] = []
    enums = infer_enum_values(spec)
    if enums:
        return ["enum_invalid"]
    if chosen_type in {"int", "float", "number", "bool"}:
        rules.append("type_widen")
    elif chosen_type == "string":
        rules.append("string_empty")
    elif chosen_type in {"list", "tuple", "sequence"}:
        rules.append("structure_shrink")
    elif chosen_type == "tensor":
        rules.extend(["shape_expand", "shape_shrink", "dtype_mutate"])
    elif chosen_type in {"dict", "dtype", "device", "any", "bytes"}:
        rules.append("type_switch")
    return list(dict.fromkeys(rules))


def build_signature_binding(params: Dict[str, ParameterRuntime], sig: RuntimeSignature) -> Dict[str, Any]:
    bound_positional: List[str] = []
    bound_keyword: List[str] = []
    skipped_optional: List[str] = []
    for name, param in params.items():
        if not param.include_in_base_call:
            skipped_optional.append(name)
            continue
        if param.call_kind in {"positional_only", "var_positional"}:
            bound_positional.append(name)
        elif param.call_kind == "var_keyword":
            bound_keyword.append(f"**{name}")
        else:
            bound_keyword.append(name)
    return {
        "bound_positional": bound_positional,
        "bound_keyword": bound_keyword,
        "skipped_optional": skipped_optional,
        "accepts_var_keyword": bool(sig.accepts_kwargs if sig else False),
    }


def build_param_dependencies(api_full_name: str, params: Dict[str, ParameterRuntime]) -> List[str]:
    api = api_full_name.lower()
    deps: List[str] = []
    for name, param in params.items():
        lname = name.lower()
        seed = param.base_seed_spec if isinstance(param.base_seed_spec, dict) else {}
        spec = param.spec if isinstance(param.spec, dict) else {}
        dtype_candidates = [str(x) for x in spec.get("dtype_candidates", []) if str(x)]
        text = " ".join([clean_text(spec.get("size", "")), clean_text(spec.get("description", ""))] + [clean_text(x) for x in spec.get("constraints", [])])
        if lname in {"shape", "dims", "size"} or lname.endswith("_shape"):
            value = seed.get("value")
            if isinstance(value, list):
                deps.append(f"{name}.rank == 1")
            if dtype_candidates:
                deps.append(f"{name}.dtype in {{{','.join(dtype_candidates)}}}")
            elif "int" in text.lower():
                deps.append(f"{name}.dtype in {{int32,int64}}")
        if lname in {"axis", "axes", "dim", "dims"}:
            deps.append(f"{name} valid for input rank")
        if lname in {"dtype", "out_type", "output_type"} or lname.endswith("_dtype"):
            deps.append(f"{name} is a supported dtype object")
    if "broadcast" in api and "shape" in params:
        data_names = [name for name in ("input", "x", "tensor", "a") if name in params]
        input_name = data_names[0] if data_names else "input"
        deps.append(f"broadcast_compatible({input_name}.shape, shape)")
    return list(dict.fromkeys(dep for dep in deps if dep))


def build_raise_contract(api_full_name: str, params: Dict[str, ParameterRuntime]) -> List[Dict[str, Any]]:
    api = api_full_name.lower()
    contracts: List[Dict[str, Any]] = []
    for name, param in params.items():
        lname = name.lower()
        neg = set(param.negative_mutation_rules or [])
        if "broadcast" in api and lname == "shape":
            contracts.append({
                "when": "negative_mutation(shape_scalar)",
                "expect": ["InvalidArgumentError", "TypeError", "ValueError"],
            })
            continue
        if "enum_invalid" in neg:
            contracts.append({"when": f"negative_mutation({name}_invalid_enum)", "expect": ["TypeError", "ValueError"]})
        elif neg and (lname in {"axis", "axes", "dim", "dims"} or "shape" in lname or "dtype" in lname):
            contracts.append({"when": f"negative_mutation({name})", "expect": ["TypeError", "ValueError", "InvalidArgumentError"]})
    return contracts


def estimate_doc_confidence(spec: Dict[str, Any], params: Dict[str, ParameterRuntime], readiness_reasons: Sequence[str]) -> float:
    if readiness_reasons:
        return 0.25
    if not params:
        return 0.5
    grounded = 0
    for param in params.values():
        spec_obj = param.spec if isinstance(param.spec, dict) else {}
        if spec_obj.get("description") or spec_obj.get("constraints") or spec_obj.get("dtype_candidates") or spec_obj.get("enum_values"):
            grounded += 1
    score = 0.65 + 0.3 * (grounded / max(1, len(params)))
    if spec.get("constraints"):
        score += 0.03
    return round(min(score, 0.98), 2)


def build_runtime_object_from_spec(api_full_name: str, spec: Dict[str, Any]) -> RuntimeAPIObject:
    module_path = clean_text(spec.get("module_path", ""))
    api_name = clean_text(spec.get("api_name", ""))
    import_path = f"{module_path}.{api_name}" if module_path and api_name else api_full_name
    backend = detect_backend(import_path, module_path)
    allow_self = needs_explicit_self(import_path)
    params_raw = spec.get("params", {}) if isinstance(spec.get("params"), dict) else {}
    sig = runtime_signature(import_path)
    if sig.found:
        filtered: Dict[str, Any] = {}
        for name, value in params_raw.items():
            if is_fake_param_name(name) and not (allow_self and clean_text(name).lower() == "self"):
                continue
            lname = clean_text(name).lower()
            if backend == "tensorflow" and lname == "name" and name not in sig.params:
                continue
            if sig.params and name not in sig.params and not sig.accepts_kwargs:
                continue
            filtered[name] = value
        for name, param in sig.params.items():
            if name in filtered:
                continue
            if backend == "tensorflow" and name == "name":
                continue
            if param.default is inspect._empty:
                filtered[name] = synthesize_spec_for_signature_param(name, param, backend)
        ordered: Dict[str, Any] = {}
        for name in sig.params:
            if name in filtered:
                ordered[name] = filtered[name]
        for name, value in filtered.items():
            if name not in ordered:
                ordered[name] = value
        params_raw = ordered
    params: Dict[str, ParameterRuntime] = {}
    readiness_reasons: List[str] = []

    for name, p in params_raw.items():
        if not isinstance(p, dict):
            readiness_reasons.append(f"param {name} is not an object")
            continue
        if jax_padding_enum_values(api_full_name, name):
            p = dict(p)
            existing = infer_enum_values(p)
            if not existing:
                p["enum_values"] = jax_padding_enum_values(api_full_name, name)
            p.setdefault("case_insensitive", True)
        if is_fake_param_name(name) and not (allow_self and clean_text(name).lower() == "self"):
            continue
        if backend == "tensorflow" and clean_text(name).lower() == "name":
            continue
        sig_param = sig.params.get(name) if sig.found else None
        runtime_required = bool(sig_param is not None and sig_param.default is inspect._empty)
        include = clean_text(p.get("flag", "")) == "Required" or runtime_required
        default_text = clean_text(p.get("default", ""))
        if not include and default_text:
            include = False
        base_seed_spec, chosen_type, unresolved = build_base_seed_spec(api_full_name, name, p, backend)
        if clean_text(p.get("flag", "")) == "Required" and unresolved:
            readiness_reasons.append(f"required param '{name}' unresolved: {unresolved}")
        valid_rules = build_valid_mutation_rules(p, chosen_type)
        negative_rules = build_negative_mutation_rules(p, chosen_type)
        params[name] = ParameterRuntime(
            name=name,
            spec=p,
            chosen_type=chosen_type,
            include_in_base_call=include,
            base_seed_spec=base_seed_spec,
            mutation_rules=valid_rules + negative_rules,
            valid_mutation_rules=valid_rules,
            negative_mutation_rules=negative_rules,
            unresolved_reason=unresolved,
            call_kind=signature_call_kind(sig_param),
        )

    if "axis" in params and "dim" in params and params["axis"].include_in_base_call and params["dim"].include_in_base_call:
        if backend == "torch":
            params["axis"].include_in_base_call = False
        else:
            params["dim"].include_in_base_call = False

    spec_ready = not readiness_reasons
    runtime_obj = RuntimeAPIObject(
        api_full_name=api_full_name,
        api_name=api_name,
        module_path=module_path,
        import_path=import_path,
        backend=backend,
        params=params,
        output=spec.get("output", {}),
        constraints=spec.get("constraints", []) if isinstance(spec.get("constraints"), list) else [],
        spec_ready=spec_ready,
        ready_for_stage4=spec_ready,
        readiness_reasons=readiness_reasons,
        signature_binding=build_signature_binding(params, sig),
        param_dependencies=build_param_dependencies(api_full_name, params),
        raise_contract=build_raise_contract(api_full_name, params),
        doc_confidence=estimate_doc_confidence(spec, params, readiness_reasons),
    )
    get_adapter(backend).adjust_runtime_object(runtime_obj)
    return runtime_obj


def _dtype_object(dtype_name: str, backend: str) -> Any:
    dtype_name = clean_text(dtype_name) or "float32"
    if backend == "tensorflow" and tf is not None:
        return getattr(tf, dtype_name, tf.float32)
    if backend == "torch" and torch is not None:
        return getattr(torch, dtype_name, torch.float32)
    if np is not None:
        return getattr(np, dtype_name, np.float32)
    return dtype_name


def resolve_python_object(seed_spec: Dict[str, Any], backend: str = "python") -> Any:
    kind = seed_spec.get("kind")
    if kind == "literal":
        value = seed_spec.get("value")
        if isinstance(value, list) and seed_spec.get("as_tuple"):
            return tuple(value)
        if isinstance(value, list) and seed_spec.get("element_kind") == "dtype":
            return [_dtype_object(v, backend) for v in value]
        return value
    if kind == "tuple":
        items = seed_spec.get("items", [])
        if not isinstance(items, list):
            return tuple()
        return tuple(
            resolve_python_object(item, backend=backend) if isinstance(item, dict) else item
            for item in items
        )
    if kind == "list":
        items = seed_spec.get("items", [])
        if not isinstance(items, list):
            return []
        return [
            resolve_python_object(item, backend=backend) if isinstance(item, dict) else item
            for item in items
        ]
    if kind == "bytes":
        return bytes(seed_spec.get("value", b""))
    if kind == "serialized_tensor":
        values = seed_spec.get("values", [1.0, 2.0])
        dtype_name = clean_text(seed_spec.get("dtype", "float32")) or "float32"
        if backend == "tensorflow" and tf is not None:
            dtype = getattr(tf, dtype_name, tf.float32)
            return tf.io.serialize_tensor(tf.constant(values, dtype=dtype))
        return json.dumps(values).encode("utf-8")
    if kind == "dtype":
        return _dtype_object(seed_spec.get("value", "float32"), backend)
    if kind == "device":
        value = clean_text(seed_spec.get("value", "cpu")) or "cpu"
        if backend == "torch" and torch is not None:
            return torch.device(value)
        return value
    if kind == "numpy_datetime64":
        if np is None:
            return seed_spec.get("value", "2020-01-01")
        return np.datetime64(seed_spec.get("value", "2020-01-01"), seed_spec.get("unit", "D"))
    if kind == "numpy_timedelta64":
        if np is None:
            return int(seed_spec.get("value", 1))
        return np.timedelta64(int(seed_spec.get("value", 1)), seed_spec.get("unit", "D"))
    if kind == "numpy_dtype_instance":
        if np is None:
            return seed_spec.get("value", "float32")
        return np.dtype(seed_spec.get("value", "float32"))
    if kind == "jax_ufunc":
        import jax.numpy as jnp  # type: ignore

        return getattr(jnp, clean_text(seed_spec.get("name", "add")) or "add")
    if kind == "jax_sharding":
        import jax  # type: ignore

        return jax.sharding.SingleDeviceSharding(jax.devices()[0])
    if kind == "jax_array_list":
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore

        shape = seed_spec.get("shape", [2])
        dtype_name = clean_text(seed_spec.get("dtype", "float32")) or "float32"
        dtype = getattr(jnp, dtype_name, jnp.float32)
        arr = jnp.ones(shape, dtype=dtype)
        return [jax.device_put(arr, jax.devices()[0])]
    if kind == "tensor":
        shape = seed_spec.get("shape", [1])
        dtype_name = clean_text(seed_spec.get("dtype", "float32")) or "float32"
        fill = seed_spec.get("fill", 1.0)
        preset = seed_spec.get("preset", "")
        values = seed_spec.get("values", None)
        if backend == "tensorflow" and tf is not None:
            dtype = getattr(tf, dtype_name, tf.float32)
            if values is not None:
                return tf.constant(values, dtype=dtype)
            if preset == "boxes":
                return tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=dtype)
            if preset == "boxes3":
                return tf.constant([[[0.0, 0.0, 1.0, 1.0]]], dtype=dtype)
            if preset == "image_size":
                return tf.constant([1, 1, 1], dtype=dtype)
            if dtype_name == "string":
                return tf.fill(shape, tf.constant(str(fill if fill != 1.0 else "a"), dtype=tf.string))
            return tf.fill(shape, tf.cast(fill, dtype))
        if backend == "torch" and torch is not None:
            dtype = getattr(torch, dtype_name, torch.float32)
            if values is not None:
                return torch.tensor(values, dtype=dtype)
            return torch.full(shape, fill, dtype=dtype)
        if np is not None:
            dtype = getattr(np, dtype_name, np.float32)
            if values is not None:
                return np.array(values, dtype=dtype)
            return np.full(shape, fill, dtype=dtype)
        return {"tensor": {"shape": shape, "dtype": dtype_name, "fill": fill}}
    if kind == "tensor_list":
        items = seed_spec.get("items")
        if not isinstance(items, list) or not items:
            items = [{"kind": "tensor", "shape": [2], "dtype": "float32", "fill": 1.0, "backend": backend}]
        return [resolve_python_object(item, backend=backend) for item in items if isinstance(item, dict)]
    if kind == "sparse_tensor":
        if backend == "tensorflow" and tf is not None:
            dtype = getattr(tf, clean_text(seed_spec.get("dtype", "float32")) or "float32", tf.float32)
            return tf.SparseTensor(
                indices=seed_spec.get("indices", [[0, 0]]),
                values=tf.constant(seed_spec.get("values", [1.0]), dtype=dtype),
                dense_shape=seed_spec.get("dense_shape", [1, 1]),
            )
        return seed_spec
    if kind == "ragged_tensor":
        if backend == "tensorflow" and tf is not None:
            dtype = getattr(tf, clean_text(seed_spec.get("dtype", "float32")) or "float32", tf.float32)
            return tf.ragged.constant(seed_spec.get("values", [[1.0], [2.0]]), dtype=dtype)
        return seed_spec
    raise ValueError(f"Unsupported seed_spec kind: {kind}")


def import_api(import_path: str) -> Any:
    parts = [p for p in import_path.split(".") if p]
    if not parts:
        raise ImportError(f"Invalid import path: {import_path}")
    last_error: Optional[BaseException] = None
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        attrs = parts[i:]
        try:
            obj = importlib.import_module(module_name)
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ImportError(f"Could not import API: {import_path}")


def instantiate_base_call(runtime_obj: RuntimeAPIObject) -> Tuple[List[Any], Dict[str, Any], List[str]]:
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    reasons: List[str] = []
    for name, param in runtime_obj.params.items():
        spec_default = clean_text(param.spec.get("default", ""))
        if not param.include_in_base_call and spec_default:
            continue
        if not param.include_in_base_call and clean_text(param.spec.get("flag", "")) == "Optional":
            continue
        if param.unresolved_reason:
            reasons.append(f"{name}: {param.unresolved_reason}")
            continue
        try:
            value = resolve_python_object(param.base_seed_spec, backend=runtime_obj.backend)
        except Exception as exc:
            reasons.append(f"{name}: failed to materialize base seed: {type(exc).__name__}: {exc}")
            continue
        call_kind = getattr(param, "call_kind", "") or "keyword"
        if call_kind == "positional_only":
            args.append(value)
        elif call_kind == "var_positional":
            if isinstance(value, (list, tuple)):
                args.extend(list(value))
            else:
                args.append(value)
        elif call_kind == "var_keyword":
            if isinstance(value, dict):
                kwargs.update(value)
            else:
                reasons.append(f"{name}: **kwargs seed did not materialize to a dict")
        else:
            kwargs[name] = value
    return args, kwargs, reasons


def classify_smoke_error(error_type: str, error_message: str) -> str:
    low = clean_text(error_message).lower()
    if error_type in {"ModuleNotFoundError", "ImportError"}:
        if "does not seem to be installed" in low or "no module named" in low:
            return "missing_dependency"
        return "import_error"
    if any(pat in low for pat in ENVIRONMENT_ERROR_PATTERNS):
        return "environment_unsupported"
    if error_type == "TypeError":
        return "spec_or_seed_error"
    return "api_runtime_error"


def run_smoke_test(runtime_obj: RuntimeAPIObject, timeout_s: float = 10.0) -> Dict[str, Any]:
    result = {
        "attempted": False,
        "success": False,
        "classification": "",
        "error_type": "",
        "error": "",
        "materialization_reasons": [],
        "result_summary": {},
        "runtime_supported_here": False,
    }
    args, kwargs, reasons = instantiate_base_call(runtime_obj)
    result["materialization_reasons"] = reasons
    if reasons:
        result["classification"] = "materialization_error"
        return result
    try:
        api = import_api(runtime_obj.import_path)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["classification"] = classify_smoke_error(type(exc).__name__, str(exc))
        return result

    result["attempted"] = True
    try:
        out = api(*args, **kwargs)
        result["success"] = True
        result["runtime_supported_here"] = True
        result["classification"] = "success"
        result["result_summary"] = summarize_python_value(out)
        return result
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["classification"] = classify_smoke_error(type(exc).__name__, str(exc))
        result["traceback"] = traceback.format_exc(limit=5)
        return result


def summarize_python_value(value: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"python_type": type(value).__name__}
    if isinstance(value, bytes):
        summary.update({"length": len(value), "preview": repr(value[:32]), "base64_preview": base64.b64encode(value[:48]).decode("ascii")})
    elif tf is not None and isinstance(value, getattr(tf, "DType", ())):
        summary.update({"dtype": value.name})
    elif tf is not None and isinstance(value, getattr(tf, "TensorShape", ())):
        summary.update({"shape": value.as_list() if value.rank is not None else None})
    elif tf is not None and isinstance(value, getattr(tf, "SparseTensor", ())):
        summary.update({
            "kind": "SparseTensor",
            "dense_shape": value.dense_shape.numpy().tolist() if hasattr(value.dense_shape, "numpy") else [],
            "indices_preview": value.indices.numpy().tolist()[:8] if hasattr(value.indices, "numpy") else [],
            "values": summarize_python_value(value.values),
        })
    elif tf is not None and isinstance(value, getattr(tf, "RaggedTensor", ())):
        try:
            bounding = value.bounding_shape().numpy().tolist()
        except Exception:
            bounding = []
        summary.update({
            "kind": "RaggedTensor",
            "ragged_rank": int(value.ragged_rank),
            "bounding_shape": bounding,
            "flat_values": summarize_python_value(value.flat_values),
        })
    elif tf is not None and tf.is_tensor(value):
        try:
            arr = value.numpy()
        except Exception as exc:
            summary.update({"shape": list(value.shape), "dtype": value.dtype.name, "preview_error": f"{type(exc).__name__}: {exc}"})
            return summary
        is_float = np is not None and getattr(arr.dtype, "kind", "") in {"f", "c"}
        preview: List[Any] = []
        try:
            flat = arr.reshape(-1)[:8]
            for item in flat:
                if isinstance(item, (bytes, bytearray)):
                    preview.append(item[:32].decode("utf-8", errors="replace"))
                elif hasattr(item, "item"):
                    scalar = item.item()
                    preview.append(repr(scalar) if isinstance(scalar, complex) else scalar)
                else:
                    preview.append(item)
        except Exception:
            preview = []
        summary.update({
            "shape": list(value.shape),
            "dtype": value.dtype.name,
            "numel": int(arr.size),
            "preview": preview,
            "has_nan": bool(np.isnan(arr).any()) if is_float else False,
            "has_inf": bool(np.isinf(arr).any()) if is_float else False,
        })
    elif torch is not None and isinstance(value, torch.Tensor):
        numeric = value.numel() and value.dtype != torch.bool
        as_float = value.float() if numeric else value
        summary.update({
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "numel": int(value.numel()),
            "has_nan": bool(torch.isnan(as_float).any().item()) if numeric else False,
            "has_inf": bool(torch.isinf(as_float).any().item()) if numeric else False,
        })
    elif np is not None and isinstance(value, np.ndarray):
        is_float = np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.complexfloating)
        summary.update({
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
            "has_nan": bool(np.isnan(value).any()) if is_float else False,
            "has_inf": bool(np.isinf(value).any()) if is_float else False,
        })
    elif isinstance(value, (tuple, list)):
        summary["length"] = len(value)
    elif isinstance(value, dict):
        summary["keys"] = list(value.keys())[:16]
    elif isinstance(value, (int, float, bool, str)):
        summary["value"] = value
    return summary


def clone_value(value: Any) -> Any:
    if tf is not None and tf.is_tensor(value):
        return tf.identity(value)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if np is not None and isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _mutate_array(value: Any, rule: str, backend: str) -> Any:
    if tf is not None and tf.is_tensor(value):
        arr = value.numpy()
        mutated = _mutate_array(arr, rule, "numpy")
        return tf.convert_to_tensor(mutated, dtype=value.dtype)
    if torch is not None and isinstance(value, torch.Tensor):
        if rule in {"value_nan", "value_posinf", "value_neginf", "value_negzero", "value_large_magnitude"}:
            out = value.clone()
            if out.numel() == 0:
                return out
            if out.dtype.is_floating_point or out.dtype.is_complex:
                flat = out.reshape(-1)
                if rule == "value_nan":
                    flat[0] = float("nan")
                elif rule == "value_posinf":
                    flat[0] = float("inf")
                elif rule == "value_neginf":
                    flat[0] = float("-inf")
                elif rule == "value_negzero":
                    flat[0] = -0.0
                elif rule == "value_large_magnitude":
                    if flat.numel() == 1:
                        flat[0] = 1.0e4
                    else:
                        pattern = torch.where(
                            torch.arange(flat.numel(), device=flat.device) % 2 == 0,
                            torch.tensor(1.0e4, device=flat.device),
                            torch.tensor(-1.0e4, device=flat.device),
                        ).to(flat.dtype)
                        flat[:] = pattern
                return out
            if rule == "value_large_magnitude" and out.dtype != torch.bool:
                info = torch.iinfo(out.dtype)
                return torch.full_like(out, min(info.max, 1000000))
            return out
        if rule == "shape_expand":
            shape = list(value.shape); shape[-1] = max(2, shape[-1] + 1)
            return torch.ones(shape, dtype=value.dtype, device=value.device)
        if rule == "shape_shrink":
            shape = list(value.shape); shape[-1] = 1
            return torch.ones(shape, dtype=value.dtype, device=value.device)
        if rule == "dtype_mutate":
            target = torch.float64 if value.dtype != torch.float64 else torch.float32
            return value.to(target)
        if rule == "value_noise":
            if value.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                return (value + 1).to(value.dtype)
            return value + torch.randn_like(value.float()) * 0.1
        if rule == "value_mask":
            out = value.clone(); out.view(-1)[::2] = 0; return out
        if rule == "value_division":
            if value.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                return torch.div(value, 2, rounding_mode="floor").to(value.dtype)
            return value / 2
    if np is not None and isinstance(value, np.ndarray):
        if rule in {"value_nan", "value_posinf", "value_neginf", "value_negzero", "value_large_magnitude"}:
            out = value.copy()
            if out.size == 0:
                return out
            if np.issubdtype(out.dtype, np.floating) or np.issubdtype(out.dtype, np.complexfloating):
                flat = out.reshape(-1)
                if rule == "value_nan":
                    flat[0] = np.nan
                elif rule == "value_posinf":
                    flat[0] = np.inf
                elif rule == "value_neginf":
                    flat[0] = -np.inf
                elif rule == "value_negzero":
                    flat[0] = -0.0
                elif rule == "value_large_magnitude":
                    if flat.size == 1:
                        flat[0] = np.asarray(1.0e4, dtype=out.dtype)
                    else:
                        pattern = np.where(np.arange(flat.size) % 2 == 0, 1.0e4, -1.0e4).astype(out.dtype, copy=False)
                        flat[:] = pattern
                return out
            if rule == "value_large_magnitude" and np.issubdtype(out.dtype, np.integer):
                info = np.iinfo(out.dtype)
                return np.full_like(out, min(info.max, 1000000))
            return out
        if rule == "shape_expand":
            shape = list(value.shape); shape[-1] = max(2, shape[-1] + 1)
            return np.ones(shape, dtype=value.dtype)
        if rule == "shape_shrink":
            shape = list(value.shape); shape[-1] = 1
            return np.ones(shape, dtype=value.dtype)
        if rule == "dtype_mutate":
            return value.astype(np.float64 if value.dtype != np.float64 else np.float32)
        if rule == "value_noise":
            if np.issubdtype(value.dtype, np.integer):
                return (value + 1).astype(value.dtype)
            return value.astype(np.float32) + np.random.normal(0, 0.1, size=value.shape).astype(np.float32)
        if rule == "value_mask":
            out = value.copy(); out.reshape(-1)[::2] = 0; return out
        if rule == "value_division":
            if np.issubdtype(value.dtype, np.integer):
                return (value // 2).astype(value.dtype)
            return value / 2
    return clone_value(value)


def _enum_values_from_runtime_meta(meta: Optional[Dict[str, Any]]) -> Tuple[List[str], bool]:
    if not isinstance(meta, dict):
        return [], False
    spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else meta
    enums = infer_enum_values(spec)
    case_insensitive = bool(spec.get("case_insensitive") or spec.get("enum_case_insensitive"))
    return enums, case_insensitive


def mutate_value(value: Any, rule: str, rng: random.Random, meta: Optional[Dict[str, Any]] = None) -> Any:
    if (tf is not None and tf.is_tensor(value)) or (torch is not None and isinstance(value, torch.Tensor)) or (np is not None and isinstance(value, np.ndarray)):
        return _mutate_array(value, rule, "array")
    if isinstance(value, bool):
        if rule in {"scalar_delta", "type_widen", "type_switch"}:
            return not value
    if isinstance(value, int) and not isinstance(value, bool):
        if rule == "scalar_delta":
            if value <= 1:
                return value + 1
            return max(1, value + rng.choice([-1, 1]))
        if rule == "type_widen":
            return float(value)
    if isinstance(value, float):
        if rule == "value_nan":
            return float("nan")
        if rule == "value_posinf":
            return float("inf")
        if rule == "value_neginf":
            return float("-inf")
        if rule == "value_negzero":
            return -0.0
        if rule == "value_large_magnitude":
            return 1.0e4
        if rule == "scalar_delta":
            return value + rng.choice([-0.5, 0.5])
        if rule == "type_widen":
            return int(value) if math.isfinite(value) else 1
    if isinstance(value, int) and rule == "value_large_magnitude" and not isinstance(value, bool):
        return 1000000
    if isinstance(value, bytes):
        if rule == "type_switch":
            return b""
    if isinstance(value, str):
        if rule == "enum_switch":
            enums, case_insensitive = _enum_values_from_runtime_meta(meta)
            if enums:
                current = value.lower() if case_insensitive else value
                candidates = [
                    item
                    for item in enums
                    if (item.lower() if case_insensitive else item) != current
                ]
                return rng.choice(candidates) if candidates else value
            return value
        if rule == "enum_invalid":
            return value + "_mut"
        if rule == "string_replace":
            return value + "_mut"
        if rule == "string_empty":
            return ""
    if isinstance(value, list):
        if rule in {"value_nan", "value_posinf", "value_neginf", "value_negzero", "value_large_magnitude"} and value:
            out = list(value)
            out[0] = mutate_value(out[0], rule, rng, meta)
            return out
        if rule == "structure_grow":
            return value + [clone_value(value[-1]) if value else 1]
        if rule == "structure_shrink":
            return value[: max(1, len(value) - 1)]
        if rule == "element_delta":
            out = list(value)
            if out and isinstance(out[0], (int, float)):
                out[0] = out[0] + 1
            return out
    if isinstance(value, tuple):
        as_list = list(value)
        if rule in {"value_nan", "value_posinf", "value_neginf", "value_negzero", "value_large_magnitude"} and as_list:
            as_list[0] = mutate_value(as_list[0], rule, rng, meta)
            return tuple(as_list)
        if rule == "structure_grow":
            as_list.append(as_list[-1] if as_list else 1)
            return tuple(as_list)
        if rule == "structure_shrink":
            return tuple(as_list[: max(1, len(as_list) - 1)])
        if rule == "element_delta":
            if as_list and isinstance(as_list[0], (int, float)):
                as_list[0] = as_list[0] + 1
            return tuple(as_list)
    if rule == "type_switch":
        if isinstance(value, (int, float, bool)):
            return "bad_type"
        if isinstance(value, str):
            return 1
    return clone_value(value)
