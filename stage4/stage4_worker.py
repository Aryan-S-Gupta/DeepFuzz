#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for candidate in [Path.cwd(), ROOT]:
    c = str(candidate)
    if c not in sys.path:
        sys.path.insert(0, c)

try:
    from coverage import Coverage
except Exception:  # pragma: no cover
    Coverage = None

from json2init.deepfuzz_common import (  # noqa: E402
    clone_value,
    import_api,
    load_json,
    mutate_value,
    resolve_python_object,
    summarize_python_value,
    write_json,
)


SIZE_RULES = {"shape_expand", "shape_shrink", "structure_grow", "structure_shrink"}
TYPE_RULES = {"dtype_mutate", "type_widen", "type_switch"}
VALUE_RULES = {
    "value_noise",
    "value_mask",
    "value_division",
    "value_nan",
    "value_posinf",
    "value_neginf",
    "value_negzero",
    "value_large_magnitude",
    "scalar_delta",
    "element_delta",
    "string_replace",
    "string_empty",
    "enum_switch",
    "noop_clone",
}
EXPECTED_NEGATIVE_ERROR_TYPES = {
    "TypeError",
    "ValueError",
    "InvalidArgumentError",
    "NotFoundError",
    "UnimplementedError",
    "OpError",
    "IndexError",
    "KeyError",
    "AssertionError",
    "RuntimeError",
}
NON_VALIDATION_FAILURE_TYPES = {"NativeCrashOrAbort", "WorkerTimeout", "SystemExit", "KeyboardInterrupt"}


def signal_name(exit_code: Any) -> str:
    try:
        code = int(exit_code)
    except Exception:
        return ""
    if code >= 0:
        return ""
    try:
        return signal.Signals(-code).name
    except Exception:
        return f"SIG{-code}"


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in str(name))


def import_backend_version(backend: str) -> str:
    try:
        if backend == "torch":
            import torch

            return str(getattr(torch, "__version__", ""))
        if backend == "tensorflow":
            import tensorflow as tf

            return str(getattr(tf, "__version__", ""))
        if backend == "jax":
            import jax

            return str(getattr(jax, "__version__", ""))
    except Exception:
        return ""
    return ""


def _call_kind(meta: Dict[str, Any]) -> str:
    return str(meta.get("call_kind", "") or meta.get("parameter_kind", "") or "keyword")


def materialize_call_details(init_obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any], List[str], Dict[str, Tuple[Any, ...]], Dict[str, Any]]:
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    reasons: List[str] = []
    locations: Dict[str, Tuple[Any, ...]] = {}
    values_by_param: Dict[str, Any] = {}
    backend = str(init_obj.get("backend", "python") or "python")
    for name, meta in (init_obj.get("params") or {}).items():
        if not isinstance(meta, dict):
            reasons.append(f"{name}: invalid runtime param object")
            continue
        include = bool(meta.get("include_in_base_call"))
        spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else {}
        default_text = str(spec.get("default", "") or "").strip()
        if not include and default_text:
            continue
        if not include and str(spec.get("flag", "") or "") == "Optional":
            continue
        unresolved = str(meta.get("unresolved_reason", "") or "").strip()
        if unresolved:
            reasons.append(f"{name}: {unresolved}")
            continue
        try:
            value = resolve_python_object(meta.get("base_seed_spec", {}), backend=backend)
        except Exception as exc:
            reasons.append(f"{name}: materialization failed: {type(exc).__name__}: {exc}")
            continue
        kind = _call_kind(meta)
        if kind == "positional_only":
            locations[name] = ("arg", len(args))
            values_by_param[name] = value
            args.append(value)
        elif kind == "var_positional":
            start = len(args)
            if isinstance(value, (list, tuple)):
                args.extend(list(value))
                count = len(value)
            else:
                args.append(value)
                count = 1
            locations[name] = ("varargs", start, count)
            values_by_param[name] = value
        elif kind == "var_keyword":
            if isinstance(value, dict):
                kwargs.update(value)
                locations[name] = ("varkwargs", tuple(value.keys()))
                values_by_param[name] = value
            else:
                reasons.append(f"{name}: **kwargs seed did not materialize to a dict")
        else:
            kwargs[name] = value
            locations[name] = ("kwarg", name)
            values_by_param[name] = value
    return args, kwargs, reasons, locations, values_by_param


def materialize_call(init_obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any], List[str]]:
    args, kwargs, reasons, _, _ = materialize_call_details(init_obj)
    return args, kwargs, reasons


def materialize_kwargs(init_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Backward-compatible helper for older tests/importers."""

    _, kwargs, reasons = materialize_call(init_obj)
    return kwargs, reasons


def execute_api(init_obj: Dict[str, Any], args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error_type": "",
        "error": "",
        "result_summary": {},
        "duration_ms": 0.0,
        "has_nan": False,
        "has_inf": False,
        "invoked_api": False,
    }
    api_path = str(init_obj.get("import_path", "") or "")
    t0 = time.perf_counter()
    try:
        api = import_api(api_path)
        out["invoked_api"] = True
        result = api(*args, **kwargs)
        result_summary = summarize_python_value(result)
        out["success"] = True
        out["result_summary"] = result_summary
        out["has_nan"] = bool(result_summary.get("has_nan", False))
        out["has_inf"] = bool(result_summary.get("has_inf", False))
    except BaseException as exc:
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)
    finally:
        out["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
    return out


def _as_numpy(value: Any) -> Any:
    try:
        import numpy as _np
    except Exception:
        _np = None  # type: ignore
    try:
        import tensorflow as _tf  # type: ignore
    except Exception:
        _tf = None  # type: ignore
    try:
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None  # type: ignore

    if _tf is not None and _tf.is_tensor(value):
        return value.numpy()
    if _torch is not None and isinstance(value, _torch.Tensor):
        return value.detach().cpu().numpy()
    if _np is not None:
        try:
            import jax  # type: ignore

            array_cls = getattr(jax, "Array", ())
            if array_cls and isinstance(value, array_cls):
                try:
                    value.block_until_ready()
                except Exception:
                    pass
                return _np.asarray(value)
        except Exception:
            pass
        if isinstance(value, _np.ndarray):
            return value
    return value


def _move_value_to_device(value: Any, backend: str, device: Dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_move_value_to_device(item, backend, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, backend, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, backend, device) for key, item in value.items()}
    if backend == "torch":
        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                return value.to(device["id"])
        except Exception:
            return value
    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore

            with tf.device(str(device["id"])):
                if tf.is_tensor(value):
                    return tf.identity(value)
                if isinstance(value, tf.SparseTensor):
                    return tf.SparseTensor(
                        indices=tf.identity(value.indices),
                        values=tf.identity(value.values),
                        dense_shape=tf.identity(value.dense_shape),
                    )
                if isinstance(value, tf.RaggedTensor):
                    return value.with_flat_values(tf.identity(value.flat_values))
        except Exception:
            return value
    if backend == "jax":
        try:
            import jax  # type: ignore

            if hasattr(value, "shape") and hasattr(value, "dtype"):
                return jax.device_put(value, device["id"])
        except Exception:
            return value
    return value


def _available_torch_devices() -> List[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception:
        return []
    devices = [{"label": "cpu", "id": "cpu", "kind": "cpu"}]
    try:
        if torch.cuda.is_available():
            devices.append({"label": "cuda:0", "id": "cuda:0", "kind": "accelerator"})
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append({"label": "mps", "id": "mps", "kind": "accelerator"})
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append({"label": "xpu:0", "id": "xpu:0", "kind": "accelerator"})
    except Exception:
        pass
    return devices


def _available_tensorflow_devices() -> List[Dict[str, Any]]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return []
    devices: List[Dict[str, Any]] = []
    try:
        logical = list(tf.config.list_logical_devices())
    except Exception:
        logical = []
    for dev in logical:
        dtype = str(getattr(dev, "device_type", "") or "").upper()
        name = str(getattr(dev, "name", "") or "")
        if dtype == "CPU" and not any(d["kind"] == "cpu" for d in devices):
            devices.append({"label": "cpu", "id": name or "/CPU:0", "kind": "cpu"})
        elif dtype in {"GPU", "TPU"}:
            devices.append({"label": dtype.lower(), "id": name or f"/{dtype}:0", "kind": "accelerator"})
    if not any(d["kind"] == "cpu" for d in devices):
        devices.insert(0, {"label": "cpu", "id": "/CPU:0", "kind": "cpu"})
    return devices


def _available_jax_devices() -> List[Dict[str, Any]]:
    try:
        import jax  # type: ignore
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        for dev in jax.devices():
            platform = str(getattr(dev, "platform", "") or "").lower()
            out.append({
                "label": f"{platform}:{getattr(dev, 'id', len(out))}",
                "id": dev,
                "kind": "cpu" if platform == "cpu" else "accelerator",
            })
    except Exception:
        return []
    return out


def _select_oracle_devices(backend: str, requested: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    if backend == "torch":
        available = _available_torch_devices()
    elif backend == "tensorflow":
        available = _available_tensorflow_devices()
    elif backend == "jax":
        available = _available_jax_devices()
    else:
        return [], f"device oracle does not support backend {backend!r}"
    if not available:
        return [], "no compatible devices discovered"

    tokens = [str(item).strip().lower() for item in requested if str(item).strip()]
    if not tokens or tokens == ["auto"]:
        tokens = ["cpu", "accelerator"]
    selected: List[Dict[str, Any]] = []
    for token in tokens:
        match: Optional[Dict[str, Any]] = None
        if token in {"gpu", "cuda", "accelerator"}:
            match = next((dev for dev in available if dev.get("kind") == "accelerator"), None)
        elif token == "cpu":
            match = next((dev for dev in available if dev.get("kind") == "cpu"), None)
        else:
            match = next((dev for dev in available if token in str(dev.get("label", "")).lower() or token in str(dev.get("id", "")).lower()), None)
        if match and all(str(match.get("label")) != str(dev.get("label")) for dev in selected):
            selected.append(match)
    if len(selected) < 2:
        labels = [str(dev.get("label", "")) for dev in available]
        return selected, f"need at least two requested devices; available={labels}"
    return selected, ""


def _sync_output(value: Any, backend: str, device: Dict[str, Any]) -> None:
    try:
        if backend == "torch":
            import torch  # type: ignore

            dev = str(device.get("id", ""))
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            elif dev.startswith("mps") and hasattr(torch, "mps"):
                torch.mps.synchronize()
            return
        if backend == "jax" and hasattr(value, "block_until_ready"):
            value.block_until_ready()
    except Exception:
        pass


def _execute_api_on_oracle_device(init_obj: Dict[str, Any], args: Sequence[Any], kwargs: Dict[str, Any], device: Dict[str, Any]) -> Dict[str, Any]:
    backend = str(init_obj.get("backend", "python") or "python")
    moved_args = [_move_value_to_device(arg, backend, device) for arg in args]
    moved_kwargs = {key: _move_value_to_device(value, backend, device) for key, value in kwargs.items()}
    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore

            with tf.device(str(device["id"])):
                out = execute_api(init_obj, moved_args, moved_kwargs)
        except BaseException as exc:
            out = {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "result_summary": {},
                "duration_ms": 0.0,
                "has_nan": False,
                "has_inf": False,
                "invoked_api": False,
            }
    else:
        out = execute_api(init_obj, moved_args, moved_kwargs)
    _sync_output(_as_numpy(out.get("result_summary")), backend, device)
    out["oracle_device"] = str(device.get("label", ""))
    return out


def _compare_arrays(left: Any, right: Any, rtol: float, atol: float) -> Tuple[bool, str]:
    import numpy as _np

    a = _np.asarray(left)
    b = _np.asarray(right)
    if tuple(a.shape) != tuple(b.shape):
        return False, f"shape mismatch: {list(a.shape)} != {list(b.shape)}"
    if str(a.dtype) != str(b.dtype):
        return False, f"dtype mismatch: {a.dtype} != {b.dtype}"
    if a.dtype.kind in {"f", "c"}:
        finite_equal = _np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not finite_equal:
            return False, "floating values differ beyond tolerance"
        try:
            zero_mask = (a == 0) & (b == 0)
            if _np.any(zero_mask) and _np.any(_np.signbit(a[zero_mask]) != _np.signbit(b[zero_mask])):
                return False, "signed zero bits differ"
        except Exception:
            pass
        return True, ""
    try:
        if not _np.array_equal(a, b):
            return False, "array values differ"
    except Exception:
        if repr(a) != repr(b):
            return False, "array values differ"
    return True, ""


def _compare_values(left: Any, right: Any, rtol: float, atol: float) -> Tuple[bool, str]:
    import numpy as _np

    left_np = _as_numpy(left)
    right_np = _as_numpy(right)
    if isinstance(left_np, _np.ndarray) or isinstance(right_np, _np.ndarray):
        return _compare_arrays(left_np, right_np, rtol, atol)
    if isinstance(left_np, (list, tuple)) and isinstance(right_np, (list, tuple)):
        if len(left_np) != len(right_np):
            return False, f"sequence length mismatch: {len(left_np)} != {len(right_np)}"
        for idx, (l_item, r_item) in enumerate(zip(left_np, right_np)):
            ok, reason = _compare_values(l_item, r_item, rtol, atol)
            if not ok:
                return False, f"[{idx}] {reason}"
        return True, ""
    if isinstance(left_np, dict) and isinstance(right_np, dict):
        if set(left_np) != set(right_np):
            return False, "dict keys differ"
        for key in sorted(left_np):
            ok, reason = _compare_values(left_np[key], right_np[key], rtol, atol)
            if not ok:
                return False, f"{key}: {reason}"
        return True, ""
    if isinstance(left_np, float) or isinstance(right_np, float):
        try:
            if _np.isclose(float(left_np), float(right_np), rtol=rtol, atol=atol, equal_nan=True):
                if float(left_np) == 0.0 and float(right_np) == 0.0 and bool(_np.signbit(left_np)) != bool(_np.signbit(right_np)):
                    return False, "signed zero bits differ"
                return True, ""
        except Exception:
            pass
        return False, f"scalar values differ: {left_np!r} != {right_np!r}"
    return (left_np == right_np, "" if left_np == right_np else f"values differ: {left_np!r} != {right_np!r}")


def _comparable_output(init_obj: Dict[str, Any], args: Sequence[Any], kwargs: Dict[str, Any], device: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    backend = str(init_obj.get("backend", "python") or "python")
    moved_args = [_move_value_to_device(arg, backend, device) for arg in args]
    moved_kwargs = {key: _move_value_to_device(value, backend, device) for key, value in kwargs.items()}
    api = import_api(str(init_obj.get("import_path", "") or ""))
    t0 = time.perf_counter()
    try:
        if backend == "tensorflow":
            import tensorflow as tf  # type: ignore

            with tf.device(str(device["id"])):
                value = api(*moved_args, **moved_kwargs)
        else:
            value = api(*moved_args, **moved_kwargs)
        _sync_output(value, backend, device)
        summary = summarize_python_value(value)
        return {
            "success": True,
            "error_type": "",
            "error": "",
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            "result_summary": summary,
            "has_nan": bool(summary.get("has_nan", False)),
            "has_inf": bool(summary.get("has_inf", False)),
            "invoked_api": True,
            "oracle_device": str(device.get("label", "")),
        }, value
    except BaseException as exc:
        return {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            "result_summary": {},
            "has_nan": False,
            "has_inf": False,
            "invoked_api": True,
            "oracle_device": str(device.get("label", "")),
        }, None


def run_device_oracle(init_obj: Dict[str, Any], args: Sequence[Any], kwargs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not config.get("enabled"):
        return {"enabled": False, "available": False, "reason": "device oracle disabled"}
    backend = str(init_obj.get("backend", "python") or "python")
    devices, reason = _select_oracle_devices(backend, config.get("devices", []))
    if reason:
        return {
            "enabled": True,
            "available": False,
            "reason": reason,
            "requested_devices": list(config.get("devices", [])),
            "devices": [str(dev.get("label", "")) for dev in devices],
        }
    rtol = float(config.get("rtol", 1e-4) or 1e-4)
    atol = float(config.get("atol", 1e-5) or 1e-5)
    executions: List[Dict[str, Any]] = []
    values: List[Any] = []
    for device in devices:
        execution, value = _comparable_output(init_obj, args, kwargs, device)
        executions.append(execution)
        values.append(value)
    reference = executions[0]
    comparisons: List[Dict[str, Any]] = []
    mismatch = False
    reason = ""
    for idx, execution in enumerate(executions[1:], start=1):
        if bool(reference.get("success")) != bool(execution.get("success")):
            mismatch = True
            reason = f"success/exception mismatch between {reference.get('oracle_device')} and {execution.get('oracle_device')}"
        elif not execution.get("success"):
            same_error = reference.get("error_type") == execution.get("error_type")
            reason = "" if same_error else "exception types differ"
            mismatch = mismatch or not same_error
        else:
            ok, cmp_reason = _compare_values(values[0], values[idx], rtol, atol)
            if not ok:
                mismatch = True
                reason = cmp_reason
        comparisons.append({
            "reference_device": reference.get("oracle_device", ""),
            "candidate_device": execution.get("oracle_device", ""),
            "equal": not mismatch if reason else True,
            "reason": reason,
            "reference_summary": reference.get("result_summary", {}),
            "candidate_summary": execution.get("result_summary", {}),
            "reference_error_type": reference.get("error_type", ""),
            "candidate_error_type": execution.get("error_type", ""),
        })
        if mismatch:
            break
    return {
        "enabled": True,
        "available": True,
        "mismatch": mismatch,
        "reason": reason,
        "devices": [str(dev.get("label", "")) for dev in devices],
        "rtol": rtol,
        "atol": atol,
        "executions": executions,
        "comparisons": comparisons,
    }


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        return repr(value)


def stable_case_hash(api: str, case_kind: str, payload: Dict[str, Any]) -> str:
    raw = "|".join([api, case_kind, _stable_json(payload)])
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _rank_of_value(value: Any) -> Optional[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return len(list(shape))
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return 1 + (_rank_of_value(value[0]) or 0)
        return 1
    return None


def _first_data_rank(kwargs: Dict[str, Any], exclude: str = "") -> int:
    for name, value in kwargs.items():
        if name == exclude:
            continue
        rank = _rank_of_value(value)
        if rank:
            return max(1, rank)
    return 1


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", "")
    if dtype:
        return str(dtype).replace("torch.", "")
    return ""


def _astype_like(value: Any, dtype_name: str) -> Any:
    try:
        if hasattr(value, "astype"):
            return value.astype(dtype_name)
        if hasattr(value, "to") and "torch" in str(type(value)):
            import torch  # type: ignore

            return value.to(getattr(torch, dtype_name, getattr(torch, "int64")))
    except Exception:
        return value
    return value


def _clip_array(value: Any, low: int, high: int) -> Any:
    try:
        import numpy as _np

        if hasattr(value, "detach"):
            import torch  # type: ignore

            return torch.clamp(value, low, high)
        if hasattr(value, "numpy") and "tensorflow" in str(type(value)):
            import tensorflow as _tf  # type: ignore

            return _tf.clip_by_value(value, low, high)
        if isinstance(value, _np.ndarray):
            return _np.clip(value, low, high)
    except Exception:
        return value
    return value


def _normalize_axes(value: Any, rank: int) -> Any:
    def clamp_axis(axis: Any) -> int:
        try:
            axis_int = int(axis)
        except Exception:
            axis_int = 0
        if axis_int < 0:
            axis_int += rank
        return min(max(axis_int, 0), max(rank - 1, 0))

    if isinstance(value, tuple):
        out: List[int] = []
        for item in value:
            axis = clamp_axis(item)
            if axis not in out:
                out.append(axis)
        return tuple(out or [0])
    if isinstance(value, list):
        out = []
        for item in value:
            axis = clamp_axis(item)
            if axis not in out:
                out.append(axis)
        return out or [0]
    return clamp_axis(value)


def _num_choices(kwargs: Dict[str, Any]) -> int:
    choices = kwargs.get("choices")
    if isinstance(choices, (list, tuple)):
        return max(1, len(choices))
    rank = _rank_of_value(choices)
    if rank:
        shape = getattr(choices, "shape", None)
        try:
            return max(1, int(list(shape)[0]))
        except Exception:
            return 1
    return 1


def validate_or_repair_valid_mutation(
    api_name: str,
    param_name: str,
    value: Any,
    kwargs: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
    rule: str,
) -> Tuple[Any, List[str]]:
    reasons: List[str] = []
    lname = param_name.lower()
    rank = _first_data_rank(kwargs, exclude=param_name)

    if lname in {"axis", "axes", "dim", "dims"}:
        repaired = _normalize_axes(value, rank)
        if repaired != value:
            reasons.append("axis_repaired_to_rank_and_unique_constraints")
        return repaired, reasons

    if lname == "kth" or ("partition" in api_name.lower() and lname in {"k", "kth"}):
        repaired = _normalize_axes(value, rank)
        if isinstance(repaired, (tuple, list)):
            repaired = repaired[0] if repaired else 0
        if repaired != value:
            reasons.append("kth_repaired_to_dimension_bounds")
        return repaired, reasons

    dtype_name = _dtype_name(value).lower()
    if "index" in lname or "indices" in lname or ("choose" in api_name.lower() and lname in {"a", "indices"}):
        if dtype_name and not any(tok in dtype_name for tok in ["int", "uint"]):
            value = _astype_like(value, "int64")
            reasons.append("index_array_repaired_to_integer_dtype")
        if "choose" in api_name.lower() and lname in {"a", "indices"}:
            value = _clip_array(value, 0, _num_choices(kwargs) - 1)
            reasons.append("choose_indices_clipped_to_choice_domain")
        return value, reasons

    if dtype_name in {"uint8", "uint8_t"} or dtype_name.endswith("uint8"):
        if rule in {"value_noise", "value_division", "value_mask", "scalar_delta", "element_delta"}:
            value = _clip_array(value, 0, 255)
            value = _astype_like(value, "uint8")
            reasons.append("uint8_dtype_preserved_for_valid_value_mutation")
        return value, reasons

    return value, reasons


def _start_child_coverage(config: Dict[str, Any], seed: int) -> Any:
    if not config.get("enabled") or Coverage is None:
        return None
    data_file = str(config.get("data_file") or "")
    if not data_file:
        return None
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    cov = Coverage(
        data_file=data_file,
        branch=True,
        source=list(config.get("source") or []) or None,
        omit=list(config.get("omit") or []) or None,
        data_suffix=True,
    )
    cov.start()
    return cov


def _stop_child_coverage(cov: Any) -> None:
    if cov is None:
        return
    try:
        cov.stop()
        cov.save()
    except Exception:
        pass


def _execute_case_worker(q: Any, init_obj: Dict[str, Any], mutation_case: Dict[str, Any], seed: int, coverage_config: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "materialization_errors": [],
        "base_kwargs_materialized": False,
        "execution": {},
    }
    cov = _start_child_coverage(coverage_config, seed)
    try:
        rng = random.Random(seed)
        args, kwargs, reasons, locations, values_by_param = materialize_call_details(init_obj)
        payload["materialization_errors"] = reasons
        if reasons:
            q.put(payload)
            return
        payload["base_kwargs_materialized"] = True
        q.put({"__event__": "materialized"})
        if mutation_case:
            name = str(mutation_case.get("param", "") or "")
            rule = str(mutation_case.get("rule", "") or "")
            location = locations.get(name)
            if not location:
                payload["execution"] = {
                    "success": False,
                    "error_type": "MaterializationError",
                    "error": f"mutation param {name} was not materialized in base call",
                    "result_summary": {},
                    "duration_ms": 0.0,
                    "has_nan": False,
                    "has_inf": False,
                    "invoked_api": False,
                }
                q.put(payload)
                return
            params = init_obj.get("params", {}) if isinstance(init_obj.get("params"), dict) else {}
            current_value = values_by_param.get(name)
            mutated_value = mutate_value(current_value, rule, rng, params.get(name) if isinstance(params.get(name), dict) else None)
            if str(mutation_case.get("mutation_intent", "") or "") == "valid":
                validation_context = dict(kwargs)
                validation_context[name] = current_value
                mutated_value, repairs = validate_or_repair_valid_mutation(
                    api_name=str(init_obj.get("api_full_name", "") or ""),
                    param_name=name,
                    value=mutated_value,
                    kwargs=validation_context,
                    meta=params.get(name) if isinstance(params.get(name), dict) else None,
                    rule=rule,
                )
                payload["post_mutation_validation"] = {"valid": True, "repairs": repairs}
            payload["mutation_value_summary"] = summarize_python_value(mutated_value)
            kind = str(location[0])
            if kind == "kwarg":
                kwargs[str(location[1])] = mutated_value
            elif kind == "arg":
                args[int(location[1])] = mutated_value
            elif kind == "varargs":
                start, count = int(location[1]), int(location[2])
                replacement = list(mutated_value) if isinstance(mutated_value, (list, tuple)) else [mutated_value]
                args[start : start + count] = replacement
            elif kind == "varkwargs":
                if isinstance(mutated_value, dict):
                    for key in location[1]:
                        kwargs.pop(str(key), None)
                    kwargs.update(mutated_value)
                else:
                    payload["execution"] = {
                        "success": False,
                        "error_type": "MaterializationError",
                        "error": f"mutation param {name} no longer materialized to **kwargs dict",
                        "result_summary": {},
                        "duration_ms": 0.0,
                        "has_nan": False,
                        "has_inf": False,
                        "invoked_api": False,
                    }
                    q.put(payload)
                    return
        payload["call_summary"] = {
            "args": [summarize_python_value(v) for v in args],
            "kwargs": {name: summarize_python_value(value) for name, value in sorted(kwargs.items())},
        }
        payload["execution"] = execute_api(init_obj, args, kwargs)
        intent = str(mutation_case.get("mutation_intent", "valid") or "valid") if mutation_case else "valid"
        device_config = coverage_config.get("device_oracle", {}) if isinstance(coverage_config.get("device_oracle"), dict) else {}
        if payload["execution"].get("success") and intent == "valid" and device_config.get("enabled"):
            oracle = run_device_oracle(init_obj, args, kwargs, device_config)
            payload["device_oracle"] = oracle
            payload["execution"]["device_oracle"] = oracle
            if oracle.get("mismatch"):
                payload["execution"]["differential_mismatch"] = True
        q.put(payload)
    except BaseException as exc:
        payload["execution"] = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "result_summary": {},
            "duration_ms": 0.0,
            "has_nan": False,
            "has_inf": False,
            "invoked_api": False,
        }
        try:
            q.put(payload)
        except Exception:
            pass
    finally:
        _stop_child_coverage(cov)


def _drain_queue(q: Any) -> Tuple[List[Dict[str, Any]], bool]:
    messages: List[Dict[str, Any]] = []
    materialized = False
    while True:
        try:
            item = q.get_nowait()
        except Exception:
            break
        if isinstance(item, dict) and item.get("__event__") == "materialized":
            materialized = True
        elif isinstance(item, dict):
            messages.append(item)
    return messages, materialized


def run_isolated_case(
    init_obj: Dict[str, Any],
    seed: int,
    timeout_sec: int,
    mutation_case: Optional[Dict[str, Any]] = None,
    coverage_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_execute_case_worker, args=(q, init_obj, mutation_case or {}, seed, coverage_config or {}))
    p.start()
    p.join(timeout_sec)
    timed_out = False
    if p.is_alive():
        timed_out = True
        p.kill()
        p.join()

    messages, materialized = _drain_queue(q)
    if messages:
        payload = messages[-1]
        payload.setdefault("base_kwargs_materialized", materialized or bool(payload.get("base_kwargs_materialized")))
        return payload

    exit_code = int(p.exitcode if p.exitcode is not None else -1)
    error_type = "WorkerTimeout" if timed_out else "NativeCrashOrAbort"
    error = (
        f"isolated execution timed out after {timeout_sec}s"
        if timed_out
        else f"isolated execution subprocess exited abnormally (exitcode={exit_code}, signal={signal_name(exit_code)})"
    )
    return {
        "materialization_errors": [],
        "base_kwargs_materialized": materialized,
        "execution": {
            "success": False,
            "error_type": error_type,
            "error": error,
            "result_summary": {},
            "duration_ms": 0.0,
            "has_nan": False,
            "has_inf": False,
            "invoked_api": materialized,
            "exit_code": exit_code,
            "signal": signal_name(exit_code),
            "timeout": timed_out,
        },
    }


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _edge_rules_for_meta(meta: Dict[str, Any]) -> List[str]:
    seed = meta.get("base_seed_spec", {}) if isinstance(meta.get("base_seed_spec"), dict) else {}
    spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else {}
    chosen = str(meta.get("chosen_type", "") or spec.get("type", "") or "").lower()
    dtype = str(seed.get("dtype", "") or spec.get("dtype", "") or "").lower()
    numeric_type = any(token in chosen for token in ["tensor", "array", "float", "int", "number", "scalar"])
    if not numeric_type and seed.get("kind") != "tensor":
        return []
    integer_like = any(token in dtype for token in ["int", "uint", "bool", "string"]) or any(
        token in chosen for token in ["int", "bool", "string"]
    )
    if integer_like:
        return ["value_large_magnitude"]
    return ["value_negzero", "value_nan", "value_posinf", "value_neginf", "value_large_magnitude"]


def choose_rules(meta: Dict[str, Any], budget: int, rng: random.Random, intent: str, edge_oracle_mutations: bool = False) -> List[str]:
    if budget <= 0:
        return []
    key = "valid_mutation_rules" if intent == "valid" else "negative_mutation_rules"
    rules = [str(x) for x in (meta.get(key) or []) if str(x)]
    if not rules and intent == "valid":
        legacy = [str(x) for x in (meta.get("mutation_rules") or []) if str(x)]
        rules = [r for r in legacy if r in VALUE_RULES]
    if not rules and intent == "negative":
        legacy = [str(x) for x in (meta.get("mutation_rules") or []) if str(x)]
        rules = [r for r in legacy if r in SIZE_RULES or r in TYPE_RULES]
    if edge_oracle_mutations and intent == "valid":
        rules = _dedupe_keep_order(_edge_rules_for_meta(meta) + rules)
    if not rules:
        return []
    triad: List[str] = []
    if intent == "valid":
        for bucket in (VALUE_RULES,):
            candidates = [r for r in rules if r in bucket]
            if candidates:
                triad.append(rng.choice(candidates))
    else:
        for bucket in (SIZE_RULES, TYPE_RULES):
            candidates = [r for r in rules if r in bucket]
            if candidates:
                triad.append(rng.choice(candidates))
    pool = list(rules)
    rng.shuffle(pool)
    ordered = _dedupe_keep_order(triad + pool)
    return ordered[:budget]


def classify_mutation_result(intent: str, exec_result: Dict[str, Any]) -> Tuple[str, bool]:
    if exec_result.get("success"):
        return ("valid_mutation_success" if intent == "valid" else "negative_accepted_not_bug"), False
    error_type = str(exec_result.get("error_type", "") or "")
    if intent == "negative" and error_type and error_type not in NON_VALIDATION_FAILURE_TYPES:
        return "expected_negative_rejection", True
    if intent == "valid" and error_type in EXPECTED_NEGATIVE_ERROR_TYPES:
        return "invalid_valid_mutation", False
    return ("valid_mutation_failure" if intent == "valid" else "negative_mutation_unexpected_failure"), False


def param_in_base_call(meta: Dict[str, Any]) -> bool:
    spec = meta.get("spec", {}) if isinstance(meta.get("spec"), dict) else {}
    if not bool(meta.get("include_in_base_call")):
        return False
    if str(meta.get("unresolved_reason", "") or "").strip():
        return False
    if str(spec.get("default", "") or "").strip() and str(spec.get("flag", "") or "") != "Required":
        return False
    if str(spec.get("flag", "") or "") == "Optional":
        return False
    return True


def build_mutated_cases(
    init_obj: Dict[str, Any],
    mutation_budget: int,
    rng: random.Random,
    case_timeout_sec: int,
    seed: int,
    coverage_config: Optional[Dict[str, Any]] = None,
    edge_oracle_mutations: bool = False,
) -> List[Dict[str, Any]]:
    mutations: List[Dict[str, Any]] = []
    params = init_obj.get("params", {}) if isinstance(init_obj.get("params"), dict) else {}
    negative_budget = max(1, min(2, mutation_budget)) if mutation_budget > 0 else 0
    for name, meta in params.items():
        if not isinstance(meta, dict) or not param_in_base_call(meta):
            continue
        for intent, budget in [("valid", mutation_budget), ("negative", negative_budget)]:
            for rule in choose_rules(meta, budget, rng, intent, edge_oracle_mutations=edge_oracle_mutations):
                mutation_case = {"param": name, "rule": rule, "mutation_intent": intent}
                case = run_isolated_case(
                    init_obj,
                    seed=seed + len(mutations) + 1,
                    timeout_sec=case_timeout_sec,
                    mutation_case=mutation_case,
                    coverage_config=coverage_config,
                )
                exec_result = case.get("execution", {}) if isinstance(case.get("execution"), dict) else {}
                if case.get("materialization_errors") and not exec_result:
                    exec_result = {
                        "success": False,
                        "error_type": "materialization_error",
                        "error": " | ".join(str(x) for x in case.get("materialization_errors", [])),
                        "result_summary": {},
                        "duration_ms": 0.0,
                        "has_nan": False,
                        "has_inf": False,
                        "invoked_api": False,
                    }
                classification, expected_failure = classify_mutation_result(intent, exec_result)
                oracle = case.get("device_oracle") if isinstance(case.get("device_oracle"), dict) else {}
                if oracle:
                    exec_result["device_oracle"] = oracle
                if intent == "valid" and oracle.get("mismatch"):
                    exec_result["differential_mismatch"] = True
                    exec_result["error_type"] = "DifferentialMismatch"
                    exec_result["error"] = str(oracle.get("reason", "device outputs differ") or "device outputs differ")
                    classification = "differential_mismatch"
                    expected_failure = False
                exec_result.update(
                    {
                        "param": name,
                        "rule": rule,
                        "mutation_intent": intent,
                        "classification": classification,
                        "expected_failure": expected_failure,
                        "mutated_value_summary": case.get("mutation_value_summary", {}),
                        "post_mutation_validation": case.get("post_mutation_validation", {}),
                    }
                )
                if intent == "valid" and exec_result.get("success"):
                    exec_result["testcase_hash"] = stable_case_hash(
                        str(init_obj.get("api_full_name", "") or ""),
                        "mutation",
                        {
                            "param": name,
                            "rule": rule,
                            "mutated_value": exec_result.get("mutated_value_summary", {}),
                            "result": exec_result.get("result_summary", {}),
                        },
                    )
                mutations.append(exec_result)
    return mutations


def fuzz_one(
    init_path: str,
    mutation_budget: int,
    seed: int,
    case_timeout_sec: int = 30,
    coverage_config: Optional[Dict[str, Any]] = None,
    edge_oracle_mutations: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    init_obj = load_json(init_path)
    api_name = str(init_obj.get("api_full_name", Path(init_path).stem.replace(".init", "")))
    backend = str(init_obj.get("backend", "python") or "python")
    result_bundle: Dict[str, Any] = {
        "api_full_name": api_name,
        "import_path": str(init_obj.get("import_path", "") or ""),
        "backend": backend,
        "library_version": import_backend_version(backend),
        "init_json_path": init_path,
        "seed": seed,
        "base_execution": {},
        "mutations": [],
        "materialization_errors": [],
        "base_kwargs_materialized": False,
        "worker_pid": os.getpid(),
    }
    if not init_obj.get("ready_for_stage4"):
        result_bundle["materialization_errors"] = list(init_obj.get("readiness_reasons", []))
        return result_bundle

    base_case = run_isolated_case(init_obj, seed=seed, timeout_sec=case_timeout_sec, coverage_config=coverage_config)
    reasons = list(base_case.get("materialization_errors", []) or [])
    result_bundle["materialization_errors"] = reasons
    result_bundle["base_kwargs_materialized"] = bool(base_case.get("base_kwargs_materialized")) and not bool(reasons)
    if reasons:
        return result_bundle

    base_result = base_case.get("execution", {}) if isinstance(base_case.get("execution"), dict) else {}
    if isinstance(base_case.get("device_oracle"), dict):
        base_result["device_oracle"] = base_case["device_oracle"]
        if base_case["device_oracle"].get("mismatch"):
            base_result["differential_mismatch"] = True
            base_result["error_type"] = "DifferentialMismatch"
            base_result["error"] = str(base_case["device_oracle"].get("reason", "device outputs differ") or "device outputs differ")
    if base_result.get("success"):
        base_result["testcase_hash"] = stable_case_hash(
            api_name,
            "base",
            {"call": base_case.get("call_summary", {}), "result": base_result.get("result_summary", {})},
        )
    result_bundle["base_execution"] = base_result
    if not base_result.get("success"):
        return result_bundle

    result_bundle["mutations"] = build_mutated_cases(
        init_obj,
        mutation_budget,
        rng,
        case_timeout_sec=case_timeout_sec,
        seed=seed,
        coverage_config=coverage_config,
        edge_oracle_mutations=edge_oracle_mutations,
    )
    valid_hashes = []
    if base_result.get("success") and base_result.get("testcase_hash"):
        valid_hashes.append(base_result["testcase_hash"])
    for mutation in result_bundle["mutations"]:
        if isinstance(mutation, dict) and mutation.get("mutation_intent") == "valid" and mutation.get("success") and mutation.get("testcase_hash"):
            valid_hashes.append(mutation["testcase_hash"])
    result_bundle["valid_testcase_hashes"] = list(dict.fromkeys(valid_hashes))
    return result_bundle


def start_python_coverage(enabled: bool, source: Sequence[str], omit: Sequence[str], data_file: str) -> Any:
    if not enabled or Coverage is None:
        return None
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    cov = Coverage(data_file=data_file, branch=True, source=list(source) or None, omit=list(omit) or None, data_suffix=True)
    cov.erase()
    cov.start()
    return cov


def stop_python_coverage(cov: Any, data_file: str, json_path: str, html_dir: str) -> Dict[str, Any]:
    if cov is None:
        return {"enabled": False}
    summary: Dict[str, Any] = {"enabled": True, "data_file": data_file}
    try:
        cov.stop()
        cov.save()
        try:
            cov.combine(data_paths=[os.path.dirname(data_file)])
            cov.save()
        except Exception as exc:
            summary["combine_error"] = f"{type(exc).__name__}: {exc}"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        cov.json_report(outfile=json_path, pretty_print=True)
        summary["json_path"] = json_path
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            totals = data.get("totals", {}) if isinstance(data, dict) else {}
            if isinstance(totals, dict):
                summary["statement_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["covered_percent"] = totals.get("covered_percent", totals.get("percent_covered", ""))
                summary["covered_lines"] = totals.get("covered_lines", "")
                summary["num_statements"] = totals.get("num_statements", "")
                nb = totals.get("num_branches", 0) or 0
                cb = totals.get("covered_branches", 0) or 0
                if nb:
                    summary["branch_percent"] = round(100.0 * float(cb) / float(nb), 4)
        except Exception as exc:
            summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        try:
            cov.html_report(directory=html_dir)
            summary["html_dir"] = html_dir
        except Exception as exc:
            summary["html_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        summary["export_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def run_worker(
    init_paths: Sequence[str],
    results_json: str,
    mutation_budget: int,
    seed: int,
    enable_python_coverage: bool = False,
    python_cov_source: Sequence[str] = (),
    python_cov_omit: Sequence[str] = (),
    python_cov_data_file: str = "",
    python_cov_json: str = "",
    python_cov_html: str = "",
    case_timeout_sec: int = 30,
    enable_device_oracle: bool = False,
    device_oracle_devices: Sequence[str] = (),
    device_oracle_rtol: float = 1e-4,
    device_oracle_atol: float = 1e-5,
    edge_oracle_mutations: bool = False,
) -> Dict[str, Any]:
    cov = start_python_coverage(enable_python_coverage, python_cov_source, python_cov_omit, python_cov_data_file) if enable_python_coverage else None
    coverage_config = {
        "enabled": bool(enable_python_coverage),
        "source": list(python_cov_source or []),
        "omit": list(python_cov_omit or []),
        "data_file": python_cov_data_file,
        "device_oracle": {
            "enabled": bool(enable_device_oracle),
            "devices": list(device_oracle_devices or []),
            "rtol": float(device_oracle_rtol),
            "atol": float(device_oracle_atol),
        },
    }
    bundles: List[Dict[str, Any]] = []
    for idx, init_path in enumerate(init_paths):
        bundles.append(
            fuzz_one(
                init_path,
                mutation_budget=mutation_budget,
                seed=seed + idx,
                case_timeout_sec=case_timeout_sec,
                coverage_config=coverage_config,
                edge_oracle_mutations=edge_oracle_mutations,
            )
        )
    py_summary = stop_python_coverage(cov, python_cov_data_file, python_cov_json, python_cov_html) if cov is not None else {"enabled": False}
    payload = {
        "worker_pid": os.getpid(),
        "seed": seed,
        "mutation_budget": mutation_budget,
        "bundles": bundles,
        "python_coverage": py_summary,
        "device_oracle": {
            "enabled": bool(enable_device_oracle),
            "devices": list(device_oracle_devices or []),
            "rtol": float(device_oracle_rtol),
            "atol": float(device_oracle_atol),
            "edge_mutations": bool(edge_oracle_mutations),
        },
    }
    write_json(results_json, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 4 isolated worker")
    ap.add_argument("--init-path", action="append", default=[])
    ap.add_argument("--init-list", default="")
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--mutation-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--enable-python-coverage", action="store_true")
    ap.add_argument("--python-cov-source", action="append", default=[])
    ap.add_argument("--python-cov-omit", action="append", default=[])
    ap.add_argument("--python-cov-data-file", default="")
    ap.add_argument("--python-cov-json", default="")
    ap.add_argument("--python-cov-html", default="")
    ap.add_argument("--case-timeout-sec", type=int, default=30)
    ap.add_argument("--enable-device-oracle", action="store_true")
    ap.add_argument("--device-oracle-device", action="append", default=[])
    ap.add_argument("--device-oracle-rtol", type=float, default=1e-4)
    ap.add_argument("--device-oracle-atol", type=float, default=1e-5)
    ap.add_argument("--edge-oracle-mutations", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    init_paths = list(args.init_path or [])
    if args.init_list:
        with open(args.init_list, "r", encoding="utf-8") as f:
            init_paths.extend([line.strip() for line in f if line.strip()])
    if not init_paths:
        raise SystemExit("stage4 worker received no init paths")
    run_worker(
        init_paths=init_paths,
        results_json=args.results_json,
        mutation_budget=args.mutation_budget,
        seed=args.seed,
        enable_python_coverage=args.enable_python_coverage,
        python_cov_source=args.python_cov_source,
        python_cov_omit=args.python_cov_omit,
        python_cov_data_file=args.python_cov_data_file,
        python_cov_json=args.python_cov_json,
        python_cov_html=args.python_cov_html,
        case_timeout_sec=args.case_timeout_sec,
        enable_device_oracle=args.enable_device_oracle,
        device_oracle_devices=args.device_oracle_device,
        device_oracle_rtol=args.device_oracle_rtol,
        device_oracle_atol=args.device_oracle_atol,
        edge_oracle_mutations=args.edge_oracle_mutations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
