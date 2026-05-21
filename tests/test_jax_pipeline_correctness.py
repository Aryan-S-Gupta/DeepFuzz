from __future__ import annotations

import csv
import json

from common.api_policy import is_internal_api
from doc2info.filter_accepted import filter_one
from json2init.deepfuzz_common import build_runtime_object_from_spec, mutate_value
from stage4 import known_bug_benchmarks


def test_jax_internal_namespace_policy_rejects_implementation_paths():
    assert is_internal_api("jax._src.numpy.lax_numpy.add")
    assert is_internal_api("jax.interpreters.mlir.ir.np.array")
    assert is_internal_api("jaxlib.xla_extension.ArrayImpl")


def test_jax_public_namespace_policy_keeps_user_facing_paths():
    assert not is_internal_api("jax.numpy.add")
    assert not is_internal_api("jax.lax.padtype_to_pads")
    assert not is_internal_api("jax.nn.relu")
    assert not is_internal_api("jax.random.PRNGKey")
    assert not is_internal_api("jax.scipy.special.logsumexp")


def test_filter_rejects_internal_jax_and_accepts_public_jax_doc():
    internal_decision, _ = filter_one({
        "api": "jax.interpreters.mlir.ir.np.array",
        "resolved": True,
        "kind": "function",
        "signature": "(x)",
        "doc": "array(x)\n\nParameters\n----------\nx : array\n    input array\nReturns\n-------\narray\n    output array",
    })
    public_decision, _ = filter_one({
        "api": "jax.numpy.add",
        "resolved": True,
        "kind": "function",
        "signature": "(x, y)",
        "doc": "add(x, y)\n\nParameters\n----------\nx : array\n    numeric tensor with dtype and shape constraints\ny : array\n    numeric tensor with dtype and shape constraints\nReturns\n-------\narray\n    numeric output array",
    })

    assert internal_decision.accepted is False
    assert "internal implementation namespace" in internal_decision.reason
    assert public_decision.accepted is True


def test_jax_padding_enum_valid_mutations_stay_in_domain():
    spec = {
        "api_name": "padtype_to_pads",
        "module_path": "jax.lax",
        "params": {
            "padding": {
                "type": "string",
                "flag": "Required",
                "description": "padding mode",
                "constraints": [],
            }
        },
        "output": {"type": "tuple", "numbers": "1", "description": "pads"},
        "constraints": [],
    }

    runtime = build_runtime_object_from_spec("jax.lax.padtype_to_pads", spec).to_dict()
    padding_meta = runtime["params"]["padding"]
    seen = {mutate_value("VALID", "enum_switch", __import__("random").Random(seed), padding_meta) for seed in range(20)}

    assert padding_meta["valid_mutation_rules"] == ["enum_switch"]
    assert padding_meta["negative_mutation_rules"] == ["enum_invalid"]
    assert "SAME" in seen
    assert "SAME_LOWER" in seen
    assert "VALID_mut" not in seen
    assert mutate_value("VALID", "enum_invalid", __import__("random").Random(0), padding_meta) == "VALID_mut"


def test_known_bug_calibration_skips_when_jax_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(known_bug_benchmarks, "_import_jax", lambda: (None, "forced unavailable"))

    payload = known_bug_benchmarks.run_known_bug_benchmarks(str(tmp_path), selected=["B1"])

    assert payload["kind"] == "known_bug_calibration_seeds"
    assert payload["results"][0]["status"] == "skipped"
    assert "forced unavailable" in payload["results"][0]["skip_reason"]
    assert (tmp_path / "known_bug_benchmarks.json").exists()
