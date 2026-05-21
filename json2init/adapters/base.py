from __future__ import annotations

import re
from typing import Any, Dict, Optional


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def constraint_text(spec: Dict[str, Any]) -> str:
    return " ".join(
        [clean_text(spec.get("type", "")), clean_text(spec.get("description", "")), clean_text(spec.get("size", ""))]
        + [clean_text(x) for x in spec.get("constraints", []) if clean_text(x)]
    ).lower()


def infer_rank(spec: Dict[str, Any], default: int = 1) -> int:
    text = constraint_text(spec)
    m = re.search(r"\b([1-8])\s*[- ]?d\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"shape\s*(?:of|:|=)?\s*[:=]?\s*\(?\s*([a-z][a-z0-9_\\{} ]+(?:[,x]\s*[a-z0-9_\\{} ]+)+)\)?", text)
    if m:
        parts = re.split(r"[,x]", m.group(1))
        return max(1, min(8, len([p for p in parts if p.strip()])))
    if re.search(r"\bmatrix|matrices|matmul|matrix multiplied\b", text):
        return 2
    if re.search(r"\bvector|1d\b", text):
        return 1
    if re.search(r"\bimage|images|height|width|channels?\b", text):
        return 4
    return default


def positive_int_sequence(rank: int) -> list[int]:
    return [2 for _ in range(max(1, min(8, rank)))]


class SeedAdapter:
    """Library extension point for deterministic seed recipes."""

    def recipe(self, api_full_name: str, param_name: str, spec: Dict[str, Any], chosen_type: str, backend: str) -> Optional[Dict[str, Any]]:
        return None

    def adjust_runtime_object(self, runtime_obj: Any) -> None:
        """Mutate seed specs after all params are available."""


class GenericAdapter(SeedAdapter):
    def recipe(self, api_full_name: str, param_name: str, spec: Dict[str, Any], chosen_type: str, backend: str) -> Optional[Dict[str, Any]]:
        pname = clean_text(param_name).lower()
        text = constraint_text(spec)

        if pname in {"axis", "axes", "dim", "dims"}:
            if "tuple" in text or pname in {"axes", "dims"}:
                return {"kind": "literal", "value": [0], "as_tuple": "tuple" in text}
            return {"kind": "literal", "value": 0}

        if pname in {"shape", "size", "output_size", "input_size"} or pname.endswith("_shape") or pname.endswith("_size"):
            rank = 2 if "image" not in text else 4
            values = positive_int_sequence(rank)
            return {"kind": "literal", "value": values, "as_tuple": chosen_type == "tuple"}

        if pname in {"min", "minimum", "minval", "min_value", "low", "a_min"}:
            return {"kind": "literal", "value": 0}
        if pname in {"max", "maximum", "maxval", "max_value", "high", "a_max"}:
            return {"kind": "literal", "value": 1}

        if "index" in pname or "indices" in pname:
            if chosen_type == "tensor":
                return {"kind": "tensor", "shape": [1], "dtype": "int64", "fill": 0, "device": "cpu", "backend": backend}
            return {"kind": "literal", "value": 0}

        if chosen_type in {"int", "number"} and (
            "kernel" in pname or "stride" in pname or "dilation" in pname or "padding" in pname
        ):
            return {"kind": "literal", "value": 1 if "padding" not in pname else 0}

        return None

    def adjust_runtime_object(self, runtime_obj: Any) -> None:
        params = getattr(runtime_obj, "params", {}) or {}
        backend = getattr(runtime_obj, "backend", "python")

        tensor_params = [p for p in params.values() if getattr(p, "chosen_type", "") == "tensor"]
        ranked = [p for p in tensor_params if isinstance(getattr(p, "base_seed_spec", None), dict)]
        max_rank = 1
        for param in ranked:
            seed = param.base_seed_spec
            max_rank = max(max_rank, len(seed.get("shape", []) or []))

        pair_names = [
            ("x", "y"),
            ("input", "other"),
            ("input", "target"),
            ("mat1", "mat2"),
        ]
        for left, right in pair_names:
            if left in params and right in params:
                a = params[left].base_seed_spec
                b = params[right].base_seed_spec
                if isinstance(a, dict) and isinstance(b, dict) and a.get("kind") == b.get("kind") == "tensor":
                    rank = max(len(a.get("shape", []) or []), len(b.get("shape", []) or []), 1)
                    shape = positive_int_sequence(rank)
                    a["shape"] = shape
                    b["shape"] = list(shape)

        for name, param in params.items():
            seed = getattr(param, "base_seed_spec", {})
            if not isinstance(seed, dict):
                continue
            spec = getattr(param, "spec", {}) or {}
            text = constraint_text(spec)
            lname = clean_text(name).lower()
            if seed.get("kind") == "tensor":
                if "index" in lname or "indices" in lname:
                    seed["dtype"] = "int64"
                    seed["fill"] = 0
                rank = infer_rank(spec, default=max_rank)
                if re.search(r"\b(conv|pool|norm|image|batch|channel|height|width)\b", text):
                    rank = max(rank, 3 if "1d" in text else 4 if ("2d" in text or "image" in text) else rank)
                seed["shape"] = positive_int_sequence(rank)
                seed.setdefault("device", "cpu")
                seed.setdefault("backend", backend)

