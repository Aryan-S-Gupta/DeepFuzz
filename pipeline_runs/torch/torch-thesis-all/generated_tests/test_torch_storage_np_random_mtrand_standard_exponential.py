from pathlib import Path

from stage4.stage4_worker import fuzz_one


def test_torch_storage_np_random_mtrand_standard_exponential_base_valid():
    bundle = fuzz_one(str(Path('/Users/aryansg/Desktop/DeepFuzz/json2init/results/torch/torch.storage.np.random.mtrand.standard_exponential.init.json')), mutation_budget=0, seed=1337, case_timeout_sec=30)
    assert not bundle.get("materialization_errors"), bundle.get("materialization_errors")
    assert bundle.get("base_execution", {}).get("success"), bundle.get("base_execution", {}).get("error")
