from pathlib import Path

from stage4.stage4_worker import fuzz_one


def test_tensorflow_image_random_flip_left_right_base_valid():
    bundle = fuzz_one(str(Path('/Users/aryansg/Desktop/DeepFuzz/json2init/results/tensorflow/tensorflow.image.random_flip_left_right.init.json')), mutation_budget=0, seed=1337, case_timeout_sec=30)
    assert not bundle.get("materialization_errors"), bundle.get("materialization_errors")
    assert bundle.get("base_execution", {}).get("success"), bundle.get("base_execution", {}).get("error")
