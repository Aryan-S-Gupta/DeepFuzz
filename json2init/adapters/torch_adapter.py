from __future__ import annotations

from typing import Any, Dict, Optional

from .base import GenericAdapter, clean_text, positive_int_sequence


class TorchAdapter(GenericAdapter):
    def recipe(self, api_full_name: str, param_name: str, spec: Dict[str, Any], chosen_type: str, backend: str) -> Optional[Dict[str, Any]]:
        pname = clean_text(param_name).lower()
        api = clean_text(api_full_name).lower()
        desc = clean_text(spec.get("description", "")).lower()

        if "pool" in api and pname == "input":
            rank = 3 if "1d" in api else 4 if "2d" in api else 5 if "3d" in api else 3
            return {"kind": "tensor", "shape": positive_int_sequence(rank), "dtype": "float32", "fill": 1.0, "device": "cpu", "backend": backend}

        if any(token in api for token in ["conv1d", "conv2d", "conv3d"]) and pname == "input":
            rank = 3 if "conv1d" in api else 4 if "conv2d" in api else 5
            return {"kind": "tensor", "shape": positive_int_sequence(rank), "dtype": "float32", "fill": 1.0, "device": "cpu", "backend": backend}

        if pname in {"kernel_size", "stride", "dilation"}:
            if "tuple" in desc:
                return {"kind": "literal", "value": [1], "as_tuple": True}
            return {"kind": "literal", "value": 1}

        if pname == "padding":
            if "tuple" in desc:
                return {"kind": "literal", "value": [0], "as_tuple": True}
            return {"kind": "literal", "value": 0}

        if pname in {"input", "mat", "mat1", "mat2", "matrix"} and any(x in api for x in ["addmm", "baddbmm", "addbmm", "mm", "matmul"]):
            return {"kind": "tensor", "shape": [2, 2], "dtype": "float32", "fill": 1.0, "device": "cpu", "backend": backend}

        if pname == "vec" or pname.endswith("vec"):
            return {"kind": "tensor", "shape": [2], "dtype": "float32", "fill": 1.0, "device": "cpu", "backend": backend}

        return super().recipe(api_full_name, param_name, spec, chosen_type, backend)

    def adjust_runtime_object(self, runtime_obj: Any) -> None:
        super().adjust_runtime_object(runtime_obj)
        api = clean_text(getattr(runtime_obj, "api_full_name", "")).lower()
        params = getattr(runtime_obj, "params", {}) or {}

        if any(x in api for x in ["addmm", "mm"]):
            for name in ["input", "mat1", "mat2"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["shape"] = [2, 2]

        if "addmv" in api:
            for name in ["input", "mat"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["shape"] = [2, 2]
            for name in ["vec", "vector"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["shape"] = [2]

