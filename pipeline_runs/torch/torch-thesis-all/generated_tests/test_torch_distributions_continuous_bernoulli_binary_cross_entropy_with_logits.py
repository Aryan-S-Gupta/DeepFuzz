from pathlib import Path

from stage4.stage4_worker import fuzz_one


def test_torch_distributions_continuous_bernoulli_binary_cross_entropy_with_logits_base_valid():
    bundle = fuzz_one(str(Path('/Users/aryansg/Desktop/DeepFuzz/json2init/results/torch/torch.distributions.continuous_bernoulli.binary_cross_entropy_with_logits.init.json')), mutation_budget=0, seed=1337, case_timeout_sec=30)
    assert not bundle.get("materialization_errors"), bundle.get("materialization_errors")
    assert bundle.get("base_execution", {}).get("success"), bundle.get("base_execution", {}).get("error")
