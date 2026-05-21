from __future__ import annotations

from typing import Any, Dict, Optional

from .base import GenericAdapter, clean_text


def _tensor(
    shape: list[int],
    dtype: str = "float32",
    fill: Any = 1.0,
    values: Any = None,
) -> Dict[str, Any]:
    seed: Dict[str, Any] = {
        "kind": "tensor",
        "shape": shape,
        "dtype": dtype,
        "fill": fill,
        "device": "cpu",
        "backend": "jax",
    }
    if values is not None:
        seed["values"] = values
    return seed


def _matrix() -> Dict[str, Any]:
    return _tensor([2, 2], values=[[2.0, 0.0], [0.0, 3.0]])


def _vector3() -> Dict[str, Any]:
    return _tensor([3], values=[1.0, 2.0, 3.0])


class JaxAdapter(GenericAdapter):
    def recipe(self, api_full_name: str, param_name: str, spec: Dict[str, Any], chosen_type: str, backend: str) -> Optional[Dict[str, Any]]:
        pname = clean_text(param_name).lower()
        api = clean_text(api_full_name)
        api_low = api.lower()

        if pname == "self" and "jax.numpy.ufunc." in api_low:
            return {"kind": "jax_ufunc", "name": "add"}
        if pname == "self" and "jax.numpy.dtype." in api_low:
            return {"kind": "numpy_dtype_instance", "value": "float32"}
        if "jax.numpy.ufunc.reduceat" in api_low and pname == "indices":
            return _tensor([1], dtype="int32", values=[0])

        if "einsum" in api_low and pname == "operands":
            return {
                "kind": "tuple",
                "items": [
                    "i,i->",
                    _tensor([2], values=[1.0, 2.0]),
                    _tensor([2], values=[3.0, 4.0]),
                ],
            }

        if "busday" in api_low or any(x in api_low for x in ["datetime_", ".isnat"]):
            if pname in {"begindates", "enddates", "dates", "arr", "x"}:
                return {"kind": "numpy_datetime64", "value": "2020-01-02", "unit": "D"}
            if pname == "dtype":
                return {"kind": "numpy_dtype_instance", "value": "datetime64[D]"}
            if pname == "offsets":
                return {"kind": "literal", "value": 1}
            if pname == "unit":
                return {"kind": "literal", "value": "D"}

        if any(x in api_low for x in ["can_cast", "issubdtype"]) and pname in {"from_", "to", "arg1", "arg2"}:
            return {"kind": "numpy_dtype_instance", "value": "float32"}
        if "isdtype" in api_low:
            if pname == "dtype":
                return {"kind": "numpy_dtype_instance", "value": "float32"}
            if pname == "kind":
                return {"kind": "literal", "value": "real floating"}

        if api_low.endswith(".bmat") and pname == "obj":
            return {"kind": "literal", "value": [[[[1]], [[2]]], [[[3]], [[4]]]]}
        if api_low.endswith(".moveaxis"):
            if pname == "a":
                return _matrix()
            if pname == "source":
                return {"kind": "literal", "value": 0}
            if pname == "destination":
                return {"kind": "literal", "value": 1}
        if api_low.endswith(".setbufsize") and pname == "size":
            return {"kind": "literal", "value": 8192}
        if api_low.endswith(".bincount") and pname in {"x", "a", "input"}:
            return _tensor([3], dtype="int32", values=[0, 1, 1])
        if "packbits" in api_low and pname in {"a", "x", "input"}:
            return _tensor([4], dtype="uint8", values=[1, 0, 1, 1])
        if "unpackbits" in api_low and pname in {"a", "x", "input"}:
            return _tensor([1], dtype="uint8", values=[3])

        if "broadcast_in_dim" in api_low:
            if pname == "operand":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "shape":
                return {"kind": "literal", "value": [2]}
            if pname == "broadcast_dimensions":
                return {"kind": "literal", "value": [0]}

        if api_low == "jax.lax.full":
            if pname == "shape":
                return {"kind": "literal", "value": [2]}
            if pname == "fill_value":
                return {"kind": "literal", "value": 1.0}
            if pname == "dtype":
                return {"kind": "dtype", "value": "float32"}

        if api_low in {"jax.lax.slice", "jax.lax.dynamic_slice"}:
            if pname == "operand":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "start_indices":
                return {"kind": "literal", "value": [0]}
            if pname == "limit_indices":
                return {"kind": "literal", "value": [1]}
            if pname in {"strides", "slice_sizes"}:
                return {"kind": "literal", "value": [1]}

        if api_low == "jax.lax.pad":
            if pname == "operand":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "padding_value":
                return {"kind": "literal", "value": 0.0}
            if pname == "padding_config":
                return {
                    "kind": "tuple",
                    "items": [{"kind": "literal", "value": [0, 0, 0], "as_tuple": True}],
                }

        if api_low in {"jax.lax.reshape", "jax.numpy.reshape"}:
            if pname in {"operand", "a", "array"}:
                return _tensor([2], values=[1.0, 2.0])
            if pname in {"new_sizes", "shape"}:
                return {"kind": "literal", "value": [2]}
            if pname == "dimensions":
                return {"kind": "literal", "value": [0]}

        if api_low == "jax.lax.tile":
            if pname == "x":
                return _tensor([2], values=[1.0, 2.0])
            if pname in {"reps", "repeat"}:
                return {"kind": "literal", "value": [2]}

        if api_low == "jax.lax.select_n":
            if pname == "which":
                return {"kind": "literal", "value": 0}
            if pname == "cases":
                return {
                    "kind": "tensor_list",
                    "items": [_tensor([2], values=[1.0, 2.0]), _tensor([2], values=[3.0, 4.0])],
                }

        if api_low.startswith("jax.lax.reduce_"):
            if pname == "operand":
                dtype = "bool" if any(x in api_low for x in ["reduce_and", "reduce_or", "reduce_xor"]) else "float32"
                values = [True, False] if dtype == "bool" else [1.0, 2.0]
                return _tensor([2], dtype=dtype, values=values)
            if pname == "axes":
                return {"kind": "literal", "value": [0]}

        if api_low == "jax.lax.ragged_dot":
            if pname == "lhs":
                return _tensor([2, 2], values=[[1.0, 2.0], [3.0, 4.0]])
            if pname == "rhs":
                return _tensor([1, 2, 2], values=[[[1.0, 0.0], [0.0, 1.0]]])
            if pname == "group_sizes":
                return _tensor([1], dtype="int32", values=[2])
            if pname == "group_offset":
                return _tensor([1], dtype="int32", values=[0])

        if api_low == "jax.nn.scaled_matmul":
            if pname == "lhs":
                return _tensor([1, 2, 4], values=[[[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]])
            if pname == "rhs":
                return _tensor([1, 3, 4], values=[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
            if pname == "lhs_scales":
                return _tensor([1, 2, 4], values=[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
            if pname == "rhs_scales":
                return _tensor([1, 3, 4], values=[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
            if pname == "preferred_element_type":
                return {"kind": "dtype", "value": "float32"}

        if api_low == "jax.make_array_from_single_device_arrays":
            if pname == "shape":
                return {"kind": "literal", "value": [2]}
            if pname == "sharding":
                return {"kind": "jax_sharding"}
            if pname == "arrays":
                return {"kind": "jax_array_list", "shape": [2], "dtype": "float32"}
            if pname == "dtype":
                return {"kind": "dtype", "value": "float32"}

        if "segment_" in api_low:
            if pname == "data":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "segment_ids":
                return _tensor([2], dtype="int32", values=[0, 0])
            if pname == "num_segments":
                return {"kind": "literal", "value": 1}

        if "histogramdd" in api_low:
            if pname == "sample":
                return _tensor([2, 2], values=[[0.0, 0.0], [1.0, 1.0]])
            if pname == "bins":
                return {"kind": "literal", "value": [2, 2]}
            if pname == "range":
                return {"kind": "literal", "value": [[0.0, 1.0], [0.0, 1.0]]}
            if pname == "weights":
                return _tensor([2], values=[1.0, 1.0])

        if api_low.endswith(".compress"):
            if pname == "condition":
                return _tensor([2], dtype="bool", values=[True, False])
            if pname in {"a", "arr", "array"}:
                return _tensor([2], values=[1.0, 2.0])
            if pname == "axis":
                return {"kind": "literal", "value": 0}

        if api_low.endswith(".clip"):
            if pname == "arr":
                return _tensor([2], values=[-1.0, 2.0])
            if pname == "min":
                return {"kind": "literal", "value": 0.0}
            if pname == "max":
                return {"kind": "literal", "value": 1.0}

        if api_low.endswith(".insert"):
            if pname == "arr":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "obj":
                return {"kind": "literal", "value": 0}
            if pname == "values":
                return {"kind": "literal", "value": 3.0}

        if api_low.endswith(".ravel_multi_index"):
            if pname == "multi_index":
                return {
                    "kind": "tuple",
                    "items": [
                        _tensor([1], dtype="int32", values=[0]),
                        _tensor([1], dtype="int32", values=[1]),
                    ],
                }
            if pname == "dims":
                return {"kind": "literal", "value": [2, 2], "as_tuple": True}

        if api_low.endswith(".select"):
            if pname == "condlist":
                return {"kind": "list", "items": [_tensor([2], dtype="bool", values=[True, False])]}
            if pname == "choicelist":
                return {"kind": "list", "items": [_tensor([2], values=[1.0, 2.0])]}

        if api_low.endswith(".where"):
            if pname == "condition":
                return _tensor([2], dtype="bool", values=[True, False])
            if pname in {"x", "y"}:
                values = [1.0, 2.0] if pname == "x" else [3.0, 4.0]
                return _tensor([2], values=values)

        if api_low.endswith(".nested_iters"):
            if pname == "op":
                return _tensor([2, 2], values=[[1.0, 2.0], [3.0, 4.0]])
            if pname == "axes":
                return {"kind": "literal", "value": [[0], [1]]}

        if api_low.endswith(".split"):
            if pname == "ary":
                return _tensor([2], values=[1.0, 2.0])
            if pname == "indices_or_sections":
                return {"kind": "literal", "value": 2}
            if pname == "axis":
                return {"kind": "literal", "value": 0}

        if any(x in api_low for x in ["argpartition", ".partition"]):
            if pname in {"a", "arr"}:
                return _tensor([2], values=[2.0, 1.0])
            if pname == "kth":
                return {"kind": "literal", "value": 0}
            if pname == "axis":
                return {"kind": "literal", "value": 0}

        if api_low.endswith(".lexsort"):
            if pname == "keys":
                return {
                    "kind": "tuple",
                    "items": [
                        _tensor([2], values=[2.0, 1.0]),
                        _tensor([2], values=[1.0, 2.0]),
                    ],
                }

        if api_low.endswith(".pad"):
            if pname in {"array", "arr"}:
                return _tensor([2], values=[1.0, 2.0])
            if pname == "pad_width":
                return {"kind": "literal", "value": [[0, 0]]}
            if pname == "mode":
                return {"kind": "literal", "value": "constant"}

        if any(x in api_low for x in [".diag_indices_from", ".diagonal", ".fliplr", ".matrix_transpose", ".trace", ".tril", ".triu", ".rot90"]):
            if pname in {"a", "arr", "array", "x", "m"}:
                return _matrix()
            if pname == "axes":
                return {"kind": "literal", "value": [0, 1], "as_tuple": True}

        if any(x in api_low for x in ["linalg.cross", ".cross"]):
            if pname in {"x1", "x2", "a", "b"}:
                return _vector3()
            if pname == "axis":
                return {"kind": "literal", "value": -1}

        if any(x in api_low for x in [".matvec", ".vecmat"]):
            if api_low.endswith(".vecmat"):
                if pname == "x1":
                    return _tensor([2], values=[1.0, 2.0])
                if pname == "x2":
                    return _matrix()
            if pname in {"x1", "a", "matrix"}:
                return _matrix()
            if pname in {"x2", "x", "vector"}:
                return _tensor([2], values=[1.0, 2.0])

        if "eigh_tridiagonal" in api_low:
            if pname == "d":
                return _tensor([2], values=[2.0, 3.0])
            if pname == "e":
                return _tensor([1], values=[0.5])
            if pname == "eigvals_only":
                return {"kind": "literal", "value": True}

        if any(x in api_low for x in [
            "linalg.det", "linalg.inv", "linalg.cond", "linalg.eig", "linalg.eigh",
            "linalg.eigvals", "linalg.eigvalsh", "linalg.slogdet", "linalg.svd",
            "linalg.svdvals", "linalg.matrix_power", "scipy.linalg.det",
            "scipy.linalg.inv", "scipy.linalg.expm", "scipy.linalg.sqrtm",
            "scipy.linalg.schur", "scipy.linalg.lu_factor", "scipy.linalg.rsf2csf",
            "lax.linalg.hessenberg", "lax.linalg.householder_product", "lax.linalg.lu",
        ]):
            if pname in {"a", "x", "input", "arr", "matrix", "t"}:
                return _matrix()
            if pname == "n":
                return {"kind": "literal", "value": 2}

        if "tensorinv" in api_low:
            if pname == "a":
                return _tensor([2, 2, 2, 2], values=[[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]], [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]])
            if pname == "ind":
                return {"kind": "literal", "value": 2}

        if "solve_sylvester" in api_low:
            if pname in {"a", "b", "c"}:
                return _matrix()

        if any(x in api_low for x in ["triangular_solve", "tensorsolve"]):
            if pname in {"a", "matrix"}:
                return _matrix()
            if pname in {"b", "rhs"}:
                return _tensor([2], values=[1.0, 1.0])
            if pname in {"c", "q"}:
                return _matrix()
            if pname == "ind":
                return {"kind": "literal", "value": 1}

        if "lax.linalg.ormqr" in api_low:
            if pname in {"a", "c"}:
                return _matrix()
            if pname in {"tau", "taus"}:
                return _tensor([2], values=[1.0, 1.0])
            if pname in {"left_side", "transpose_a"}:
                return {"kind": "literal", "value": True}

        if "lax.linalg.householder_product" in api_low:
            if pname == "a":
                return _matrix()
            if pname in {"tau", "taus"}:
                return _tensor([2], values=[1.0, 1.0])

        return super().recipe(api_full_name, param_name, spec, chosen_type, backend)

    def adjust_runtime_object(self, runtime_obj: Any) -> None:
        super().adjust_runtime_object(runtime_obj)
        api = clean_text(getattr(runtime_obj, "api_full_name", "")).lower()
        params = getattr(runtime_obj, "params", {}) or {}

        if "jax.numpy.ufunc." in api or "jax.numpy.dtype." in api:
            if "self" in params:
                params["self"].include_in_base_call = True
                params["self"].call_kind = "positional_only"

        for name in ("arr", "condition", "x", "y"):
            if api.endswith(".clip") and name == "arr" and name in params:
                params[name].include_in_base_call = True
                params[name].call_kind = "positional_only"
            if api.endswith(".where") and name in params:
                params[name].include_in_base_call = True
                params[name].call_kind = "positional_only"

        if any(api.endswith(f".{name}") for name in ["unique_all", "unique_counts", "unique_inverse", "unique_values"]):
            if "x" in params:
                params["x"].call_kind = "positional_only"

        if api.endswith(".pad"):
            for name in ("array", "pad_width"):
                if name in params:
                    params[name].include_in_base_call = True

        if "eigh_tridiagonal" in api and "eigvals_only" in params:
            params["eigvals_only"].include_in_base_call = True

        if api == "jax.lax.select_n":
            if "which" in params:
                params["which"].call_kind = "positional_only"
            if "cases" in params:
                params["cases"].include_in_base_call = True
                params["cases"].call_kind = "var_positional"
