from __future__ import annotations

import csv

from common.pipeline_contract import read_api_records, write_api_csv
from common.model_config import check_model_backend
from common.result_io import discover_libraries


def test_csv_contract_reads_old_two_column_csv(tmp_path):
    path = tmp_path / "accepted.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["api_full_name", "api_doc_text"])
        writer.writeheader()
        writer.writerow({"api_full_name": "pkg.fn", "api_doc_text": "fn(x)\nArgs:\n  x: value"})

    rows = read_api_records(str(path))

    assert rows[0]["api_full_name"] == "pkg.fn"
    assert rows[0]["api_doc_text"].startswith("fn(x)")
    assert rows[0]["signature"] == ""


def test_csv_contract_writes_exactly_two_columns_and_embeds_signature(tmp_path):
    path = tmp_path / "accepted.csv"
    write_api_csv(
        str(path),
        [
            {
                "api_full_name": "pkg.fn",
                "api_doc_text": "doc",
                "signature": "(x)",
                "kind": "function",
                "module": "pkg",
                "canonical_target": "pkg.fn",
            }
        ],
    )

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == ["api_full_name", "api_doc_text"]

    rows = read_api_records(str(path))

    assert rows[0]["signature"] == "(x)"
    assert rows[0]["api_doc_text"].startswith("Observed signature: (x)")


def test_model_health_check_fails_without_model_or_supported_backend():
    ok, message = check_model_backend("", "http://localhost:11434")
    assert ok is False
    assert "no model" in message

    ok, message = check_model_backend("model", "http://localhost:11434", backend="unknown")
    assert ok is False
    assert "unsupported" in message


def test_discover_libraries_from_result_directories(tmp_path):
    for path in [
        tmp_path / "info2json" / "results" / "torch",
        tmp_path / "json_validator" / "results" / "jax",
        tmp_path / "json2init" / "results" / "tensorflow",
        tmp_path / "stage4" / "results" / "tensorflow-coverage",
    ]:
        path.mkdir(parents=True)

    assert discover_libraries(tmp_path) == ["jax", "tensorflow", "torch"]
