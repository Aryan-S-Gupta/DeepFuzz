from __future__ import annotations

import base64
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
]

# 1x1 PNG bytes, valid for TensorFlow image decoders.
_VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)

@dataclass
class ParameterRuntime:
    name: str
    spec: Dict[str, Any]
    chosen_type: str
    include_in_base_call: bool
    base_seed_spec: Dict[str, Any]
    mutation_rules: List[str]
    unresolved_reason: str = ""


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(data), f, ensure_ascii=False, indent=2)


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
        elif p in {"tensor", "bytetensor", "ndarray", "array", "sparsetensor", "raggedtensor", "tensorlike"}:
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
    return "python"


def choose_seed_type(type_str: str, description: str = "", param_name: str = "") -> str:
    tokens = parse_type_tokens(type_str)
    if not tokens:
        tokens = parse_type_tokens(description)
    pname = clean_text(param_name).lower()
    dlow = clean_text(description).lower()
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
    vals = spec.get("enum_values", [])
    if isinstance(vals, list):
        clean = [clean_text(x) for x in vals if clean_text(x)]
        if clean:
            return clean
    desc = clean_text(spec.get("description", ""))
    found = re.findall(r"[`'\"]([A-Za-z0-9_.-]+)[`'\"]", desc)
    if len(found) >= 2:
        return list(dict.fromkeys(found))
    return []


def _tf_recipe(api_full_name: str, param_name: str, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pname = param_name.lower()
    desc = clean_text(spec.get("description", "")).lower()
    # image decode style APIs
    if pname in {"contents", "serialized"} and ("decode_" in api_full_name or "image file" in desc or "serialized tensor" in desc):
        return {"kind": "bytes", "value": _VALID_PNG_BYTES}
    if pname in {"boxes"}:
        return {"kind": "tensor", "shape": [1, 4], "dtype": "float32", "fill": 0.5, "backend": "tensorflow", "preset": "boxes"}
    if pname in {"box_indices"}:
        return {"kind": "tensor", "shape": [1], "dtype": "int32", "fill": 0, "backend": "tensorflow"}
    if pname in {"crop_size"}:
        return {"kind": "literal", "value": [1, 1], "as_tuple": False}
    if pname in {"images", "image"} and api_full_name.startswith("tensorflow.image"):
        # 4-D images unless docs imply scalar bytes/encoded input.
        return {"kind": "tensor", "shape": [1, 1, 1, 3], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
    if pname in {"bounding_boxes", "boxes"} and "bounding_box" in api_full_name:
        return {"kind": "tensor", "shape": [1, 1, 4], "dtype": "float32", "fill": 0.5, "backend": "tensorflow", "preset": "boxes3"}
    if pname == "image_size":
        return {"kind": "tensor", "shape": [3], "dtype": "int32", "fill": 1, "backend": "tensorflow", "preset": "image_size"}
    if pname == "field_names":
        return {"kind": "literal", "value": ["field"], "as_tuple": False}
    if pname == "output_types":
        return {"kind": "literal", "value": ["float32"], "as_tuple": False, "element_kind": "dtype"}
    if pname == "out_type":
        return {"kind": "dtype", "value": "float32"}
    if pname in {"context_features", "sequence_features"}:
        return {"kind": "literal", "value": {}, "as_tuple": False}
    return None


def build_base_seed_spec(api_full_name: str, param_name: str, spec: Dict[str, Any], backend: str) -> Tuple[Dict[str, Any], str, str]:
    chosen = choose_seed_type(spec.get("type", ""), spec.get("description", ""), param_name)
    desc = clean_text(spec.get("description", "")).lower()
    constraint_text = " ".join(clean_text(x) for x in spec.get("constraints", [])).lower()

    if backend == "tensorflow":
        recipe = _tf_recipe(api_full_name, param_name, spec)
        if recipe is not None:
            return recipe, chosen, ""

    if chosen in UNSUPPORTED_TOKENS:
        return {}, chosen, f"unsupported base seed type: {chosen}"
    if chosen == "bool":
        return {"kind": "literal", "value": True}, chosen, ""
    if chosen == "int":
        return {"kind": "literal", "value": infer_numeric_base(spec, as_float=False)}, chosen, ""
    if chosen in {"float", "number"}:
        return {"kind": "literal", "value": infer_numeric_base(spec, as_float=True)}, chosen, ""
    if chosen == "string":
        enums = infer_enum_values(spec)
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
    rules: List[str] = []
    enums = infer_enum_values(spec)
    if enums:
        rules.append("enum_switch")
    if chosen_type in {"int", "float", "number", "bool"}:
        rules.extend(["scalar_delta", "type_widen"])
    elif chosen_type == "string":
        rules.extend(["string_replace", "string_empty"])
    elif chosen_type in {"list", "tuple", "sequence"}:
        rules.extend(["structure_grow", "structure_shrink", "element_delta"])
    elif chosen_type == "tensor":
        rules.extend(["shape_expand", "shape_shrink", "dtype_mutate", "value_noise", "value_mask", "value_division"])
    elif chosen_type in {"dict", "dtype", "device", "any", "bytes"}:
        rules.append("type_switch")
    return list(dict.fromkeys(rules))


def build_runtime_object_from_spec(api_full_name: str, spec: Dict[str, Any]) -> RuntimeAPIObject:
    module_path = clean_text(spec.get("module_path", ""))
    api_name = clean_text(spec.get("api_name", ""))
    import_path = f"{module_path}.{api_name}" if module_path and api_name else api_full_name
    backend = detect_backend(import_path, module_path)
    params_raw = spec.get("params", {}) if isinstance(spec.get("params"), dict) else {}
    params: Dict[str, ParameterRuntime] = {}
    readiness_reasons: List[str] = []

    for name, p in params_raw.items():
        if not isinstance(p, dict):
            readiness_reasons.append(f"param {name} is not an object")
            continue
        include = clean_text(p.get("flag", "")) == "Required"
        default_text = clean_text(p.get("default", ""))
        if not include and default_text:
            include = False
        base_seed_spec, chosen_type, unresolved = build_base_seed_spec(api_full_name, name, p, backend)
        if clean_text(p.get("flag", "")) == "Required" and unresolved:
            readiness_reasons.append(f"required param '{name}' unresolved: {unresolved}")
        params[name] = ParameterRuntime(
            name=name,
            spec=p,
            chosen_type=chosen_type,
            include_in_base_call=include or clean_text(p.get("flag", "")) == "Required",
            base_seed_spec=base_seed_spec,
            mutation_rules=build_mutation_rules(p, chosen_type),
            unresolved_reason=unresolved,
        )

    spec_ready = not readiness_reasons
    return RuntimeAPIObject(
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
    )


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
    if kind == "bytes":
        return bytes(seed_spec.get("value", b""))
    if kind == "dtype":
        return _dtype_object(seed_spec.get("value", "float32"), backend)
    if kind == "device":
        value = clean_text(seed_spec.get("value", "cpu")) or "cpu"
        if backend == "torch" and torch is not None:
            return torch.device(value)
        return value
    if kind == "tensor":
        shape = seed_spec.get("shape", [1])
        dtype_name = clean_text(seed_spec.get("dtype", "float32")) or "float32"
        fill = seed_spec.get("fill", 1.0)
        preset = seed_spec.get("preset", "")
        if backend == "tensorflow" and tf is not None:
            dtype = getattr(tf, dtype_name, tf.float32)
            if preset == "boxes":
                return tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=dtype)
            if preset == "boxes3":
                return tf.constant([[[0.0, 0.0, 1.0, 1.0]]], dtype=dtype)
            if preset == "image_size":
                return tf.constant([1, 1, 1], dtype=dtype)
            return tf.fill(shape, tf.cast(fill, dtype))
        if backend == "torch" and torch is not None:
            dtype = getattr(torch, dtype_name, torch.float32)
            return torch.full(shape, fill, dtype=dtype)
        if np is not None:
            dtype = getattr(np, dtype_name, np.float32)
            return np.full(shape, fill, dtype=dtype)
        return {"tensor": {"shape": shape, "dtype": dtype_name, "fill": fill}}
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


def instantiate_base_call(runtime_obj: RuntimeAPIObject) -> Tuple[Dict[str, Any], List[str]]:
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
            kwargs[name] = resolve_python_object(param.base_seed_spec, backend=runtime_obj.backend)
        except Exception as exc:
            reasons.append(f"{name}: failed to materialize base seed: {type(exc).__name__}: {exc}")
    return kwargs, reasons


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
    kwargs, reasons = instantiate_base_call(runtime_obj)
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
        out = api(**kwargs)
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
    if tf is not None and tf.is_tensor(value):
        arr = value.numpy()
        summary.update({
            "shape": list(value.shape),
            "dtype": value.dtype.name,
            "numel": int(arr.size),
            "has_nan": bool(np.isnan(arr).any()) if np is not None and arr.dtype.kind == "f" else False,
        })
    elif torch is not None and isinstance(value, torch.Tensor):
        summary.update({
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "numel": int(value.numel()),
            "has_nan": bool(torch.isnan(value.float()).any().item()) if value.numel() and value.dtype != torch.bool else False,
        })
    elif np is not None and isinstance(value, np.ndarray):
        summary.update({
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
            "has_nan": bool(np.isnan(value).any()) if np.issubdtype(value.dtype, np.floating) else False,
        })
    elif isinstance(value, (tuple, list)):
        summary["length"] = len(value)
    elif isinstance(value, dict):
        summary["keys"] = list(value.keys())[:16]
    elif isinstance(value, (int, float, bool, str, bytes)):
        summary["value"] = value if not isinstance(value, bytes) else f"<bytes:{len(value)}>"
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
            return value + torch.randn_like(value.float()) * 0.1
        if rule == "value_mask":
            out = value.clone(); out.view(-1)[::2] = 0; return out
        if rule == "value_division":
            return value / 2
    if np is not None and isinstance(value, np.ndarray):
        if rule == "shape_expand":
            shape = list(value.shape); shape[-1] = max(2, shape[-1] + 1)
            return np.ones(shape, dtype=value.dtype)
        if rule == "shape_shrink":
            shape = list(value.shape); shape[-1] = 1
            return np.ones(shape, dtype=value.dtype)
        if rule == "dtype_mutate":
            return value.astype(np.float64 if value.dtype != np.float64 else np.float32)
        if rule == "value_noise":
            return value.astype(np.float32) + np.random.normal(0, 0.1, size=value.shape).astype(np.float32)
        if rule == "value_mask":
            out = value.copy(); out.reshape(-1)[::2] = 0; return out
        if rule == "value_division":
            return value / 2
    return clone_value(value)


def mutate_value(value: Any, rule: str, rng: random.Random) -> Any:
    if (tf is not None and tf.is_tensor(value)) or (torch is not None and isinstance(value, torch.Tensor)) or (np is not None and isinstance(value, np.ndarray)):
        return _mutate_array(value, rule, "array")
    if isinstance(value, bool):
        if rule in {"scalar_delta", "type_widen", "type_switch"}:
            return not value
    if isinstance(value, int) and not isinstance(value, bool):
        if rule == "scalar_delta":
            return value + rng.choice([-1, 1])
        if rule == "type_widen":
            return float(value)
    if isinstance(value, float):
        if rule == "scalar_delta":
            return value + rng.choice([-0.5, 0.5])
        if rule == "type_widen":
            return int(value) if math.isfinite(value) else 1
    if isinstance(value, bytes):
        if rule == "type_switch":
            return b""
    if isinstance(value, str):
        if rule == "string_replace":
            return value + "_mut"
        if rule == "string_empty":
            return ""
        if rule == "enum_switch":
            return value
    if isinstance(value, list):
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
