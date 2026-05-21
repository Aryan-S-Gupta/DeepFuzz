# DeepFuzz Live Demo Run

This folder contains a small, repeatable DeepFuzz campaign for a live thesis demo.

The default API list contains one JAX API, one TensorFlow API, and one PyTorch API that reproduces a candidate bug row in the saved demo report. See `bug_demo_notes.md` for the bug explanation and upstream context.

Run the default three-API demo:

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/latest --force --mutation-budget 32 --enable-device-oracle
```

Run the same demo with selected-function Python line coverage:

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/selected-function --force --mutation-budget 32 --enable-device-oracle --coverage-scope python
```

Use `--coverage-scope both` if you want both API execution coverage and selected-function Python line coverage in the same run:

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/selected-function-both --force --mutation-budget 32 --enable-device-oracle --coverage-scope both
```

Type API names manually instead:

```bash
.venv/bin/python demo-run/run_demo.py --interactive --out demo-run/results/manual --force --mutation-budget 32 --enable-device-oracle
```

Example manual input:

```text
jax.numpy.sin
tensorflow.nn.relu
torch.masked.argmax
```

The run writes all demo artefacts under `demo-run/results/<run-id>/`:

- `all_apis.txt`: the selected API names.
- `<lib>/stage0_selection/`: manual API list and accepted documentation rows.
- `<lib>/stage1_specs/`: documentation-derived JSON specs reused from the final thesis artefacts, or generated if `--generate-missing-stage1` is used.
- `<lib>/stage2_validator/`: validation `ok.csv` and `errors.csv`.
- `<lib>/stage3_init/`: executable `.init.json` seeds plus Stage 3 readiness CSV files.
- `<lib>/generated_tests/`: replayable pytest tests.
- `<lib>/stage4_results/`: isolated fuzz results, coverage reports, execution summaries, and bug/triage reports.
- `demo_report.md` and `demo_report.json`: compact summary for presentation.
- `commands.log`: exact commands used for reproducibility.

Use `mutation-budget 32` when you want the two PyTorch bug-bearing APIs to exercise the edge numeric mutations that appear in the final thesis run. Use a smaller mutation budget only when you want a very quick smoke demonstration.

The default demo uses `--coverage-scope api_only` so TensorFlow and PyTorch do not spend live-demo time exporting whole-package HTML coverage. To run the slower selected-function coverage path on a small subset, add `--coverage-scope python`.
