# DeepFuzz Bug Demo Notes

## Curated Demo APIs

The live demo API list is intentionally small:

| Library | API | Purpose |
| --- | --- | --- |
| JAX | `jax.numpy.sin` | Simple documentation-to-test happy path. |
| TensorFlow | `tensorflow.nn.relu` | Simple documentation-to-test happy path. |
| PyTorch | `torch.masked.argmax` | Bug-bearing API from the final DeepFuzz Stage 4 report. |

This gives one API from each target library while keeping one clear bug example for triage.

## Selected-Function Coverage Command

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/selected-function --force --mutation-budget 32 --enable-device-oracle --coverage-scope python
```

Use `--coverage-scope both` if you want the report to include both API execution coverage and selected-function Python line coverage:

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/selected-function-both --force --mutation-budget 32 --enable-device-oracle --coverage-scope both
```

## Bug: `torch.masked.argmax`

DeepFuzz final run evidence:

- Final report row: `stage4/results/torch-thesis-all-coverage/bug_report.csv`
- Category: `differential_mismatch`
- Verdict used by DeepFuzz: `candidate_implementation_bug`
- Reproducer artifact: `stage4/results/torch-thesis-all-coverage/execution_results/torch.masked.argmax.results.json`
- Trigger found by DeepFuzz: a valid edge-value mutation placed `NaN` into the input tensor, then the device oracle compared CPU and accelerator output.
- Observed problem in the DeepFuzz result: CPU and Apple MPS both execute successfully, but the outputs differ beyond tolerance.

Online/upstream evidence:

- PyTorch issue `#166401`: <https://github.com/pytorch/pytorch/issues/166401>
- The issue reports that `torch.masked.argmax` returns different results on CPU vs CUDA when multiple maximum values exist.
- The issue is open and labelled `module: masked operators` and `triaged`.

How to explain it:

DeepFuzz did not start from the upstream issue. It generated a valid executable seed from documentation, mutated a tensor edge case, ran the same API through a CPU-vs-accelerator oracle, and recorded a differential mismatch. Afterward, the online check showed that `torch.masked.argmax` already has an upstream-triaged device-consistency issue, so this is the strongest demo example.

Report confidence:

High. The exact API has an upstream-triaged device inconsistency. A new report would still need a minimal reproducer in the current target version and target backend because DeepFuzz found CPU vs MPS, while the upstream issue describes CPU vs CUDA.

## How DeepFuzz Found These Bugs

1. Stage 1 converted documentation for each API into a structured JSON spec.
2. Stage 2 validated the JSON spec against schema and runtime evidence.
3. Stage 3 turned the validated spec into deterministic executable seed JSON.
4. Stage 4 ran the seed in isolation, then generated valid mutations.
5. Edge numeric mutations introduced cases such as `NaN`, infinities, signed zero, and large magnitudes.
6. The device oracle executed the same API on CPU and accelerator when available.
7. If both executions succeeded but outputs differed beyond tolerance, DeepFuzz recorded `differential_mismatch`.
8. The triage layer wrote a candidate bug row instead of treating every failure as a confirmed library bug.

## What To Say In The Demo

"This is a candidate implementation bug, not an automatic upstream-confirmed bug. The important point is that DeepFuzz found a CPU-vs-accelerator differential mismatch from documentation-derived executable tests. For `torch.masked.argmax`, I can also point to an upstream-triaged issue for the same API and the same class of device inconsistency, so this is a strong bug-demo example."
