from __future__ import annotations

import csv

from info2json.info2json import normalize_schema
from info2json.info2json import process_file


def test_normalization_uses_signature_column_when_doc_has_no_signature():
    doc = """Compute along a dimension.

Args:
  input (Tensor): input tensor.
  dim (int): dimension to reduce.

Returns:
  Tensor: output tensor.
"""
    spec = {
        "params": {
            "input": {"type": "Tensor", "description": "input tensor", "flag": "Required"},
            "dim": {"type": "int", "description": "dimension to reduce", "flag": "Optional"},
        },
        "output": {"type": "Tensor", "description": "output tensor"},
        "constraints": [],
    }

    normalized = normalize_schema(spec, "torch.fake_reduce", doc, signature="(input, dim=None)")

    assert list(normalized["params"]) == ["input", "dim"]
    assert normalized["params"]["dim"]["flag"] == "Optional"
    assert normalized["params"]["dim"]["default"] == "None"


def test_tensorflow_normalization_rejects_prose_machine_fields_and_name():
    doc = """logical_xor(x, y, name=None)

Args:
  x (Tensor): first tensor.
  y (Tensor): second tensor.
  name: A name for this operation (optional).

Returns:
  Tensor: output tensor.
"""
    spec = {
        "params": {
            "x": {"type": "Tensor", "size": "the direction in which values are read", "flag": "Required", "description": "first tensor"},
            "y": {"type": "Tensor", "flag": "Required", "description": "second tensor"},
            "name": {"type": "str", "default": "a name for this operation", "flag": "Optional", "description": "operation name"},
            "Caution": {"type": "if true", "flag": "Required", "description": "not a parameter"},
        },
        "output": {"type": "Tensor", "description": "output"},
        "constraints": [],
    }

    normalized = normalize_schema(spec, "tensorflow.math.logical_xor", doc, signature="(x, y, name=None)")

    assert list(normalized["params"]) == ["x", "y"]
    assert normalized["params"]["x"]["size"] == ""
    assert "name" not in normalized["params"]
    assert "Caution" not in normalized["params"]


def test_tensorflow_doc_examples_do_not_leak_prose_into_size_default_type():
    doc = """gather_nd(params, indices, batch_dims=0, name=None)

Args:
  params (Tensor): source tensor.
  indices (Tensor): index tensor.
  batch_dims (int): must be non-negative.
  name: A name for the operation (optional).
"""
    spec = {
        "params": {
            "params": {"type": "Tensor", "flag": "Required", "description": "source tensor"},
            "indices": {"type": "Tensor", "size": "if true, bad indices are ignored on CPU", "flag": "Required", "description": "index tensor"},
            "batch_dims": {"type": "int", "default": "defaults to 0. the leading dimensions are batch dimensions", "flag": "Optional", "description": "batch dims"},
        },
        "output": {"type": "Tensor", "description": ""},
        "constraints": [],
    }

    normalized = normalize_schema(spec, "tensorflow.gather_nd", doc, signature="(params, indices, batch_dims=0, name=None)")

    assert normalized["params"]["indices"]["size"] == ""
    assert normalized["params"]["batch_dims"]["default"] == "0"


def test_info2json_model_unavailable_records_each_pending_api(tmp_path):
    accepted = tmp_path / "accepted.csv"
    with accepted.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["api_full_name", "api_doc_text"])
        writer.writeheader()
        writer.writerow({"api_full_name": "pkg.a", "api_doc_text": "a()\nReturns:\n  int: value"})
        writer.writerow({"api_full_name": "pkg.b", "api_doc_text": "b()\nReturns:\n  int: value"})

    process_file(
        input_path=str(accepted),
        outdir=str(tmp_path / "out"),
        model="",
        host="http://localhost:11434",
        retries=1,
    )

    rows = list(csv.DictReader((tmp_path / "out" / "failures.csv").open(encoding="utf-8", newline="")))
    assert [row["api_full_name"] for row in rows] == ["pkg.a", "pkg.b"]
    assert {row["classification"] for row in rows} == {"environment_model_unavailable"}
