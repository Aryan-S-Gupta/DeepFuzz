from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from .base import GenericAdapter, clean_text, positive_int_sequence

_VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
_VALID_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQ"
    "AAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
    "9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/"
    "ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEA"
    "AgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//E"
    "ABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)
_VALID_GIF_BYTES = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
_VALID_BMP_BYTES = (
    b"BM"
    + (70).to_bytes(4, "little")
    + b"\x00\x00\x00\x00"
    + (54).to_bytes(4, "little")
    + (40).to_bytes(4, "little")
    + (2).to_bytes(4, "little", signed=True)
    + (2).to_bytes(4, "little", signed=True)
    + (1).to_bytes(2, "little")
    + (24).to_bytes(2, "little")
    + (0).to_bytes(4, "little")
    + (16).to_bytes(4, "little")
    + (2835).to_bytes(4, "little", signed=True)
    + (2835).to_bytes(4, "little", signed=True)
    + (0).to_bytes(4, "little")
    + (0).to_bytes(4, "little")
    + b"\xff\x00\x00\x00\xff\x00\x00\x00"
    + b"\x00\x00\xff\xff\xff\xff\x00\x00"
)


class TensorFlowAdapter(GenericAdapter):
    def recipe(self, api_full_name: str, param_name: str, spec: Dict[str, Any], chosen_type: str, backend: str) -> Optional[Dict[str, Any]]:
        pname = clean_text(param_name).lower()
        api = clean_text(api_full_name)
        api_low = api.lower()
        desc = clean_text(spec.get("description", "")).lower()

        if pname == "name":
            return {"kind": "literal", "value": ""}
        if pname in {"keepdims", "exclusive", "reverse", "canonicalized_coordinates", "narrow_range", "sorted"}:
            return {"kind": "literal", "value": False}
        if pname == "side":
            return {"kind": "literal", "value": "left"}
        if pname == "direction":
            return {"kind": "literal", "value": "ASCENDING"}
        if pname == "padding":
            return {"kind": "literal", "value": "VALID"}
        if pname == "data_format":
            return {"kind": "literal", "value": "NHWC"}
        if ("conv" in api_low or "pool" in api_low) and pname in {"strides", "stride"}:
            return {"kind": "literal", "value": [1, 1] if "keras.ops" in api_low else [1, 1, 1, 1], "as_tuple": False}
        if ("conv" in api_low or "pool" in api_low) and pname in {"pool_size", "kernel_size"}:
            return {"kind": "literal", "value": [1, 1], "as_tuple": False}
        if "conv" in api_low and pname in {"filters", "kernel"}:
            return {"kind": "tensor", "shape": [1, 1, 2, 2], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
        if pname == "axis":
            return {"kind": "literal", "value": -1 if any(x in api_low for x in ["crossentropy", "softmax", "sort"]) else 0}
        if pname in {"alpha", "beta", "gamma", "threshold"}:
            return {"kind": "literal", "value": 0.5}
        if pname == "epsilon":
            return {"kind": "literal", "value": 1e-7}
        if pname in {"dtype", "out_type", "output_type"} or pname.endswith("_dtype"):
            value = "string" if "string" in api_low else "int32" if any(x in api_low for x in ["arg", "count", "shape"]) else "float32"
            return {"kind": "dtype", "value": value}
        if pname == "num_bits":
            return {"kind": "literal", "value": 8}
        if pname == "block_size":
            return {"kind": "literal", "value": 2}
        if pname in {"k", "depth", "num", "count", "window_length", "sequence_length", "sequence_stride", "fft_length", "num_buckets"}:
            return {"kind": "literal", "value": 2 if pname not in {"k", "sequence_stride"} else 1}
        if "bitcast" in api_low and pname == "type":
            return {"kind": "dtype", "value": "int32"}
        if pname in {"t_list", "tensor_list", "inputs"} and "global_norm" in api_low:
            return {
                "kind": "tensor_list",
                "items": [
                    {"kind": "tensor", "shape": [2], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"},
                    {"kind": "tensor", "shape": [2], "dtype": "float32", "fill": 2.0, "backend": "tensorflow"},
                ],
            }
        if "boolean_mask" in api_low:
            if pname in {"tensor", "input", "x"}:
                return {"kind": "tensor", "shape": [2], "dtype": "float32", "values": [1.0, 2.0], "backend": "tensorflow"}
            if pname == "mask":
                return {"kind": "tensor", "shape": [2], "dtype": "bool", "values": [True, False], "backend": "tensorflow"}
        if any(x in api_low for x in ["batch_to_space", "space_to_batch", "space_to_depth", "depth_to_space"]):
            if pname in {"input", "x"}:
                if "depth_to_space" in api_low:
                    return {"kind": "tensor", "shape": [1, 2, 2, 4], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
                return {"kind": "tensor", "shape": [1, 2, 2, 1], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
            if pname in {"block_shape", "crops"}:
                return {"kind": "literal", "value": [1, 1], "as_tuple": False}
            if pname in {"paddings"}:
                return {"kind": "literal", "value": [[0, 0], [0, 0]], "as_tuple": False}
            if pname == "block_size":
                return {"kind": "literal", "value": 2}

        if "fake_quant_with_min_max_vars_per_channel_gradient" in api_low:
            if pname in {"gradients", "gradient", "grads", "inputs", "input"}:
                return {"kind": "tensor", "shape": [2, 3], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
            if pname == "min":
                return {"kind": "tensor", "shape": [3], "dtype": "float32", "values": [0.0, 0.0, 0.0], "backend": "tensorflow"}
            if pname == "max":
                return {"kind": "tensor", "shape": [3], "dtype": "float32", "values": [6.0, 6.0, 6.0], "backend": "tensorflow"}
        if "fake_quant_with_min_max_vars_per_channel" in api_low:
            if pname in {"inputs", "input"}:
                return {"kind": "tensor", "shape": [2, 3], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
            if pname == "min":
                return {"kind": "tensor", "shape": [3], "dtype": "float32", "values": [0.0, 0.0, 0.0], "backend": "tensorflow"}
            if pname == "max":
                return {"kind": "tensor", "shape": [3], "dtype": "float32", "values": [6.0, 6.0, 6.0], "backend": "tensorflow"}

        if pname == "serialized" and ("parse_tensor" in api_low or "serialized tensor" in desc):
            return {"kind": "serialized_tensor", "dtype": "float32", "values": [1.0, 2.0], "backend": "tensorflow"}
        if pname in {"contents", "serialized"} and ("decode_" in api or "image file" in desc or "serialized tensor" in desc):
            if "jpeg" in api_low or "jpg" in api_low or "extract_jpeg" in api_low:
                return {"kind": "bytes", "value": _VALID_JPEG_BYTES}
            if "bmp" in api_low:
                return {"kind": "bytes", "value": _VALID_BMP_BYTES}
            if "gif" in api_low:
                return {"kind": "bytes", "value": _VALID_GIF_BYTES}
            return {"kind": "bytes", "value": _VALID_PNG_BYTES}
        if "strings." in api_low or ".io.decode_base64" in api_low or ".io.encode_base64" in api_low or api_low.endswith(".as_string"):
            if pname in {"input", "inputs", "x", "source", "string_tensor"}:
                if "to_number" in api_low:
                    return {"kind": "tensor", "shape": [2], "dtype": "string", "values": ["1.0", "2.0"], "backend": "tensorflow"}
                return {"kind": "tensor", "shape": [2], "dtype": "string", "values": ["a", "b"], "backend": "tensorflow"}
            if pname == "pattern":
                return {"kind": "literal", "value": "a.*"}
            if pname == "rewrite":
                return {"kind": "literal", "value": "x"}
        if "draw_bounding_boxes" in api_low and pname in {"boxes", "bounding_boxes"}:
            return {"kind": "tensor", "shape": [1, 1, 4], "dtype": "float32", "fill": 0.5, "backend": "tensorflow", "preset": "boxes3"}
        if pname in {"boxes"}:
            return {"kind": "tensor", "shape": [1, 4], "dtype": "float32", "fill": 0.5, "backend": "tensorflow", "preset": "boxes"}
        if pname in {"box_indices"}:
            return {"kind": "tensor", "shape": [1], "dtype": "int32", "fill": 0, "backend": "tensorflow"}
        if pname in {"crop_size"}:
            return {"kind": "literal", "value": [1, 1], "as_tuple": False}
        if pname in {"images", "image"} and api.startswith("tensorflow.") and ("image" in api_low or "encode_" in api_low):
            if "encode_" in api_low:
                return {"kind": "tensor", "shape": [2, 2, 3], "dtype": "uint8", "fill": 1, "backend": "tensorflow"}
            if "sobel_edges" in api_low:
                return {"kind": "tensor", "shape": [1, 4, 4, 1], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
            return {"kind": "tensor", "shape": [1, 2, 2, 3], "dtype": "float32", "fill": 1.0, "backend": "tensorflow"}
        if pname in {"bounding_boxes", "boxes"} and "bounding_box" in api:
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
        if pname in {"shape", "dims", "size"} or pname.endswith("_shape"):
            return {"kind": "literal", "value": [2, 2], "as_tuple": False}
        if pname in {"paddings"}:
            return {"kind": "literal", "value": [[0, 0], [0, 0]], "as_tuple": False}
        if pname in {"value"} and any(x in api_low for x in [".fill", ".constant"]):
            return {"kind": "literal", "value": 1.0}
        if "decode_predictions" in api_low and pname in {"preds", "predictions"}:
            return {"kind": "tensor", "shape": [1, 1000], "dtype": "float32", "fill": 0.001, "backend": "tensorflow"}
        if any(x in api_low for x in ["categorical_crossentropy", "binary_crossentropy", ".losses.", ".metrics."]):
            if pname in {"y_true", "target", "targets"}:
                return {"kind": "tensor", "shape": [1, 3], "dtype": "float32", "values": [[1.0, 0.0, 0.0]], "backend": "tensorflow"}
            if pname in {"y_pred", "output", "outputs", "predictions"}:
                return {"kind": "tensor", "shape": [1, 3], "dtype": "float32", "values": [[0.9, 0.05, 0.05]], "backend": "tensorflow"}
        if any(x in api_low for x in ["cholesky", "det", "inv", "qr", "svd", "eig", "eigh"]):
            if pname in {"input", "tensor", "x", "a"}:
                return {"kind": "tensor", "shape": [2, 2], "dtype": "float32", "values": [[2.0, 0.0], [0.0, 2.0]], "backend": "tensorflow"}
        if any(x in api_low for x in ["solve", "lstsq", "triangular_solve"]):
            if pname in {"matrix", "a"}:
                return {"kind": "tensor", "shape": [2, 2], "dtype": "float32", "values": [[2.0, 0.0], [0.0, 2.0]], "backend": "tensorflow"}
            if pname in {"rhs", "b"}:
                return {"kind": "tensor", "shape": [2, 1], "dtype": "float32", "values": [[1.0], [1.0]], "backend": "tensorflow"}
        if "broadcast" in api_low and pname in {"shape", "shape_x", "shape_y"}:
            return {"kind": "tensor", "shape": [2], "dtype": "int32", "values": [2, 2], "backend": "tensorflow"}
        if "segment" in api_low and pname in {"segment_ids", "segments"}:
            return {"kind": "tensor", "shape": [2], "dtype": "int32", "values": [0, 0], "backend": "tensorflow"}
        if any(x in api_low for x in ["argmax", "argmin", "top_k", "sort"]):
            if pname in {"input", "x", "values"}:
                return {"kind": "tensor", "shape": [2], "dtype": "float32", "values": [1.0, 2.0], "backend": "tensorflow"}
        if any(x in api_low for x in ["reduce_", "reduce.", "reduce"]) and pname in {"input_tensor", "input", "x"}:
            return {"kind": "tensor", "shape": [2], "dtype": "float32", "values": [1.0, 2.0], "backend": "tensorflow"}
        if any(x in api_low for x in ["fft2d", "ifft2d"]):
            if pname in {"input", "x"}:
                return {"kind": "tensor", "shape": [2, 2], "dtype": "complex64", "fill": 1.0, "backend": "tensorflow"}
        if any(x in api_low for x in ["fft3d", "ifft3d"]):
            if pname in {"input", "x"}:
                return {"kind": "tensor", "shape": [2, 2, 2], "dtype": "complex64", "fill": 1.0, "backend": "tensorflow"}
        if "sparse." in api_low and pname in {"sp_input", "input", "x", "a"}:
            return {"kind": "sparse_tensor", "indices": [[0, 0]], "values": [1.0], "dense_shape": [1, 1], "dtype": "float32"}
        if "ragged." in api_low and pname in {"input", "inputs", "x", "rt"}:
            return {"kind": "ragged_tensor", "values": [[1.0], [2.0]], "dtype": "float32"}

        return super().recipe(api_full_name, param_name, spec, chosen_type, backend)

    def adjust_runtime_object(self, runtime_obj: Any) -> None:
        super().adjust_runtime_object(runtime_obj)
        api = clean_text(getattr(runtime_obj, "api_full_name", "")).lower()
        params = getattr(runtime_obj, "params", {}) or {}
        if "conv" in api or "pool" in api:
            for name in ["input", "inputs", "x", "image", "images"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["shape"] = positive_int_sequence(4)
        if "encode_" in api:
            for name in ["image", "images"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["shape"] = [2, 2, 3]
                    params[name].base_seed_spec["dtype"] = "uint8"
                    params[name].base_seed_spec["fill"] = 1
        if any(x in api for x in ["categorical_crossentropy", "binary_crossentropy", ".losses.", ".metrics."]):
            for name, values in [("y_true", [[1.0, 0.0, 0.0]]), ("target", [[1.0, 0.0, 0.0]]), ("targets", [[1.0, 0.0, 0.0]])]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [1, 3], "dtype": "float32", "values": values})
            for name, values in [("y_pred", [[0.9, 0.05, 0.05]]), ("output", [[0.9, 0.05, 0.05]]), ("outputs", [[0.9, 0.05, 0.05]]), ("predictions", [[0.9, 0.05, 0.05]])]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [1, 3], "dtype": "float32", "values": values})
        if "fake_quant_with_min_max_vars_per_channel_gradient" in api:
            for name in ["gradients", "gradient", "grads", "inputs", "input"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [2, 3], "dtype": "float32", "fill": 1.0})
            for name, values in [("min", [0.0, 0.0, 0.0]), ("max", [6.0, 6.0, 6.0])]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [3], "dtype": "float32", "values": values})
        elif "fake_quant_with_min_max_vars_per_channel" in api:
            for name in ["inputs", "input"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [2, 3], "dtype": "float32", "fill": 1.0})
            for name, values in [("min", [0.0, 0.0, 0.0]), ("max", [6.0, 6.0, 6.0])]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec.update({"shape": [3], "dtype": "float32", "values": values})
        if any(x in api for x in ["bitwise", "left_shift", "right_shift"]):
            for name in ["x", "y", "input"]:
                if name in params and params[name].base_seed_spec.get("kind") == "tensor":
                    params[name].base_seed_spec["dtype"] = "int32"
                    params[name].base_seed_spec["fill"] = 1
