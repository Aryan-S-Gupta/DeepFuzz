from __future__ import annotations

import pytest

from json2init.deepfuzz_common import build_runtime_object_from_spec, instantiate_base_call, summarize_python_value
from json2init.adapters.tensorflow_adapter import TensorFlowAdapter


def test_torch_matrix_seed_shapes_are_ranked():
    spec = {
        "api_name": "addmm",
        "module_path": "torch",
        "params": {
            "input": {"type": "tensor", "flag": "Required", "description": "matrix to be added"},
            "mat1": {"type": "tensor", "flag": "Required", "description": "the first matrix"},
            "mat2": {"type": "tensor", "flag": "Required", "description": "the second matrix"},
        },
        "output": {"type": "tensor", "description": ""},
        "constraints": [],
    }

    runtime = build_runtime_object_from_spec("torch.addmm", spec)

    assert runtime.params["input"].base_seed_spec["shape"] == [2, 2]
    assert runtime.params["mat1"].base_seed_spec["shape"] == [2, 2]
    assert runtime.params["mat2"].base_seed_spec["shape"] == [2, 2]


def test_positional_only_args_are_materialized_positionally():
    api = "tests.fixtures.fake_stage4_api.positional_only_api"
    spec = {
        "api_name": "positional_only_api",
        "module_path": "tests.fixtures.fake_stage4_api",
        "params": {"x": {"type": "int", "flag": "Required", "description": "value"}},
        "output": {"type": "int", "description": ""},
        "constraints": [],
    }

    runtime = build_runtime_object_from_spec(api, spec)
    args, kwargs, reasons = instantiate_base_call(runtime)

    assert reasons == []
    assert args == [1]
    assert "x" not in kwargs
    assert runtime.params["x"].call_kind == "positional_only"


def test_keyword_only_args_remain_kwargs():
    api = "tests.fixtures.fake_stage4_api.keyword_only_api"
    spec = {
        "api_name": "keyword_only_api",
        "module_path": "tests.fixtures.fake_stage4_api",
        "params": {"x": {"type": "int", "flag": "Required", "description": "value"}},
        "output": {"type": "int", "description": ""},
        "constraints": [],
    }

    runtime = build_runtime_object_from_spec(api, spec)
    args, kwargs, reasons = instantiate_base_call(runtime)

    assert reasons == []
    assert args == []
    assert kwargs["x"] == 1
    assert runtime.params["x"].call_kind == "keyword_only"


def test_pool_kernel_size_is_integer_tuple_seed():
    spec = {
        "api_name": "avg_pool1d",
        "module_path": "torch",
        "params": {
            "input": {"type": "tensor", "flag": "Required", "description": "input tensor of shape (N, C, W)"},
            "kernel_size": {
                "type": "tuple|number",
                "flag": "Required",
                "description": "the size of the window. Can be a single number or a tuple `(kW,)`",
            },
        },
        "output": {"type": "tensor", "description": ""},
        "constraints": [],
    }

    runtime = build_runtime_object_from_spec("torch.avg_pool1d", spec)

    assert runtime.params["input"].base_seed_spec["shape"] == [2, 2, 2]
    assert runtime.params["kernel_size"].base_seed_spec == {"kind": "literal", "value": [1], "as_tuple": True}


def _tf_runtime(api_name, params):
    spec = {
        "api_name": api_name.rsplit(".", 1)[-1],
        "module_path": api_name.rsplit(".", 1)[0],
        "params": params,
        "output": {"type": "tensor", "description": ""},
        "constraints": [],
    }
    return build_runtime_object_from_spec(api_name, spec)


def test_tensorflow_seed_rules_for_representative_families():
    runtime = _tf_runtime(
        "tensorflow.linalg.cholesky",
        {"input": {"type": "tensor", "flag": "Required", "description": "matrix"}},
    )
    assert runtime.params["input"].base_seed_spec["shape"] == [2, 2]

    runtime = _tf_runtime(
        "tensorflow.image.encode_png",
        {"image": {"type": "tensor", "flag": "Required", "description": "image"}},
    )
    assert runtime.params["image"].base_seed_spec["dtype"] == "uint8"
    assert runtime.params["image"].base_seed_spec["shape"] == [2, 2, 3]

    runtime = _tf_runtime(
        "tensorflow.strings.lower",
        {"input": {"type": "tensor", "flag": "Required", "description": "string tensor"}},
    )
    assert runtime.params["input"].base_seed_spec["dtype"] == "string"

    runtime = _tf_runtime(
        "tensorflow.keras.losses.categorical_crossentropy",
        {
            "y_true": {"type": "tensor", "flag": "Required", "description": "target"},
            "y_pred": {"type": "tensor", "flag": "Required", "description": "prediction"},
            "axis": {"type": "int", "flag": "Optional", "description": "axis"},
        },
    )
    assert runtime.params["y_true"].base_seed_spec["shape"] == [1, 3]
    assert runtime.params["y_pred"].base_seed_spec["shape"] == [1, 3]
    assert runtime.params["axis"].base_seed_spec["value"] == -1

    runtime = _tf_runtime(
        "tensorflow.broadcast_to",
        {
            "input": {"type": "tensor", "flag": "Required", "description": "input tensor"},
            "shape": {"type": "list", "flag": "Required", "description": "shape vector"},
        },
    )
    assert runtime.params["shape"].base_seed_spec["value"] == [2, 2]

    runtime = _tf_runtime(
        "tensorflow.bitcast",
        {
            "input": {"type": "tensor", "flag": "Required", "description": "input tensor"},
            "type": {"type": "dtype", "flag": "Required", "description": "output dtype"},
        },
    )
    assert runtime.params["type"].base_seed_spec == {"kind": "dtype", "value": "int32"}

    runtime = _tf_runtime(
        "tensorflow.boolean_mask",
        {
            "tensor": {"type": "tensor", "flag": "Required", "description": "input tensor"},
            "mask": {"type": "tensor", "flag": "Required", "description": "boolean mask"},
        },
    )
    assert runtime.params["mask"].base_seed_spec["shape"] == [2]
    assert runtime.params["mask"].base_seed_spec["dtype"] == "bool"


def test_tensorflow_fake_quant_per_channel_rules_are_shape_safe():
    runtime = _tf_runtime(
        "tensorflow.quantization.fake_quant_with_min_max_vars_per_channel",
        {
            "inputs": {"type": "tensor", "flag": "Required", "description": "input"},
            "min": {"type": "tensor", "flag": "Required", "description": "min"},
            "max": {"type": "tensor", "flag": "Required", "description": "max"},
            "num_bits": {"type": "int", "flag": "Optional", "description": "bits"},
            "narrow_range": {"type": "bool", "flag": "Optional", "description": "whether narrow"},
        },
    )

    assert runtime.params["inputs"].base_seed_spec["shape"] == [2, 3]
    assert runtime.params["min"].base_seed_spec["shape"] == [3]
    assert runtime.params["max"].base_seed_spec["values"] == [6.0, 6.0, 6.0]
    assert runtime.params["num_bits"].base_seed_spec["value"] == 8
    assert runtime.params["narrow_range"].base_seed_spec["value"] is False


def test_tensorflow_decode_and_draw_seed_recipes_are_format_specific():
    adapter = TensorFlowAdapter()

    jpeg = adapter.recipe("tensorflow.image.decode_jpeg", "contents", {"description": "encoded image file"}, "bytes", "tensorflow")
    bmp = adapter.recipe("tensorflow.io.decode_bmp", "contents", {"description": "encoded image file"}, "bytes", "tensorflow")
    gif = adapter.recipe("tensorflow.image.decode_gif", "contents", {"description": "encoded image file"}, "bytes", "tensorflow")
    serialized = adapter.recipe("tensorflow.io.parse_tensor", "serialized", {"description": "serialized tensor"}, "bytes", "tensorflow")
    boxes = adapter.recipe("tensorflow.image.draw_bounding_boxes", "boxes", {"description": "boxes"}, "tensor", "tensorflow")

    assert jpeg["value"].startswith(b"\xff\xd8")
    assert bmp["value"].startswith(b"BM")
    assert gif["value"].startswith(b"GIF")
    assert serialized["kind"] == "serialized_tensor"
    assert boxes["shape"] == [1, 1, 4]


def test_tensorflow_safe_serializer_handles_composites_when_available():
    tf = pytest.importorskip("tensorflow")

    tensor_summary = summarize_python_value(tf.constant([1.0]))
    sparse_summary = summarize_python_value(tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[1, 1]))
    ragged_summary = summarize_python_value(tf.ragged.constant([[1.0], [2.0, 3.0]]))
    dtype_summary = summarize_python_value(tf.float32)
    shape_summary = summarize_python_value(tf.TensorShape([2, 3]))
    bytes_summary = summarize_python_value(b"abc")

    assert tensor_summary["shape"] == [1]
    assert sparse_summary["kind"] == "SparseTensor"
    assert ragged_summary["kind"] == "RaggedTensor"
    assert dtype_summary["dtype"] == "float32"
    assert shape_summary["shape"] == [2, 3]
    assert bytes_summary["length"] == 3
