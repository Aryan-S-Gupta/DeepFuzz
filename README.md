# DeepFuzz

DeepFuzz is an end-to-end thesis pipeline for document-guided fuzzing of deep-learning APIs. It collects public API documentation, extracts structured API contracts with an LLM, repairs and validates those contracts, materializes executable seeds, runs isolated mutation campaigns, measures coverage, and writes strict bug/documentation-mismatch triage reports.

The reviewed repository contains the actual pipeline code plus saved run artifacts for JAX, TensorFlow, and PyTorch. Non-code review material and generated replay exports are not part of the repository.

## Repository Layout

- `doc2info/`: API collection, documentation/signature extraction, and accepted/rejected API filtering.
- `info2json/`: LLM-driven conversion from accepted documentation rows to structured JSON specs.
- `json_validator/`: schema validation and repair for generated API specs.
- `json2init/`: dependency-aware seed materialization from validated specs.
- `stage4/`: isolated execution, mutation, coverage collection, device oracle support, and bug triage.
- `scripts/`: orchestration, reporting, repair, coverage reading, and run-manifest utilities.
- `pipeline_runs/`: frozen API lists, final reports, triage tables, coverage summaries, and reproducibility manifests for saved thesis runs.
- `stage4/results/`: saved Stage 4 execution, coverage, and bug-report artifacts.

## Setup

Use Python 3.11 for the local thesis environment.

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
```

Pull the local Ollama models used by generation and repair.

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen3-coder:30b
```

Export model settings before generation, validation, or repair.

```bash
export DEEPFUZZ_MODEL_BACKEND=ollama
export OLLAMA_HOST=http://localhost:11434
export DEEPFUZZ_MODEL=qwen2.5:14b
export DEEPFUZZ_FAST_MODEL=qwen2.5-coder:7b
export DEEPFUZZ_REPAIR_MODEL=qwen2.5-coder:14b
export DEEPFUZZ_STRONG_REPAIR_MODEL=qwen3-coder:30b
```

## End-To-End Run

Replace `<lib>` with `jax`, `tensorflow`, or `torch`.

Expected outputs:

- `pipeline_runs/<lib>/<run_id>/api_list.txt`
- `pipeline_runs/<lib>/<run_id>/final_report.md`
- `pipeline_runs/<lib>/<run_id>/final_report.json`
- `stage4/results/<lib>-thesis-all-coverage/coverage_report.json`
- `stage4/results/<lib>-thesis-all-coverage/bug_report.csv`

```bash
source .venv311/bin/activate
python3 scripts/select_api_subset.py --lib <lib> --all --limit 0 --ok-csv json2init/results/<lib>/ok.csv --out pipeline_runs/<lib>/<lib>-thesis-all/api_list.txt
RUN_ID=<lib>-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/<lib>/<lib>-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/<lib>-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=<lib> MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/<lib> SPEC_DIR=info2json/results/<lib> VALIDATOR_OK_CSV=json_validator/results/<lib>/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh <lib>
python3 scripts/read_coverage.py --stage4-results-dir stage4/results/<lib>-thesis-all-coverage
```

Saved run examples:

```bash
JAX_PLATFORMS=cpu RUN_ID=jax-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/jax/jax-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/jax-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=jax MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/jax SPEC_DIR=info2json/results/jax VALIDATOR_OK_CSV=json_validator/results/jax/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh jax
RUN_ID=tensorflow-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/tensorflow/tensorflow-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/tensorflow-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=tensorflow MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/tensorflow SPEC_DIR=info2json/results/tensorflow VALIDATOR_OK_CSV=json_validator/results/tensorflow/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh tensorflow
RUN_ID=torch-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/torch/torch-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/torch-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=torch MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/torch SPEC_DIR=info2json/results/torch VALIDATOR_OK_CSV=json_validator/results/torch/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh torch
```

## Pipeline Stages

API collection:

```bash
python3 doc2info/collector.py --root <root> --out doc2info/<lib>_apis.jsonl --max-depth 5 --exclude-prefix <prefix>
```

API filtering:

```bash
python3 doc2info/filter_accepted.py --in doc2info/<lib>_apis.jsonl --outdir doc2info/results --lib-name <lib> --max 0 --max-examples 50 --allow-receiver-apis --allow-wrapper-bridges
```

Stage 1, documentation to JSON specs:

```bash
python3 info2json/info2json.py --input doc2info/results/<lib>/accepted.csv --outdir info2json/results/<lib> --model "$DEEPFUZZ_MODEL" --host "$OLLAMA_HOST" --timeout 300 --limit 0 --retries 3 --sleep 0 --overwrite --combined-out info2json/results/<lib>/combined.json --num-predict 320 --num-ctx 2048 --temperature 0 --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt
```

Stage 2, spec validation and repair:

```bash
python3 json_validator/json_validator.py --spec-dir info2json/results/<lib> --api-csv doc2info/results/<lib>/accepted.csv --state-dir json_validator/results/<lib> --primary-repair-model "$DEEPFUZZ_REPAIR_MODEL" --fallback-repair-model "$DEEPFUZZ_STRONG_REPAIR_MODEL" --fallback-after-round 3 --repair-host "$OLLAMA_HOST" --max-rounds 3 --repair-timeout 300 --repair-num-predict 700 --repair-num-ctx 4096 --repair-temperature 0 --only-apis pipeline_runs/<lib>/<run_id>/api_list.txt --force --external-errors-csv pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.csv --external-errors-jsonl pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.jsonl
```

Stage 3, spec to executable seeds:

```bash
python3 json2init/json2init.py --spec-dir info2json/results/<lib> --outdir json2init/results/<lib> --ok-csv json_validator/results/<lib>/ok.csv --overwrite --reuse-existing --limit 0 --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt --non-strict-smoke
```

Stage 4, isolated fuzzing and coverage:

```bash
python3 stage4/stage4_coverage_runner.py --init-dir json2init/results/<lib> --results-dir stage4/results/<lib>-thesis-all-coverage --ok-csv json2init/results/<lib>/ok.csv --mutation-budget 8 --seed 1337 --run-id <run_id> --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt --limit 0 --coverage-scope python --low-coverage-threshold 60 --worker-timeout-sec 120 --case-timeout-sec 30 --enable-python-coverage --python-cov-source <lib> --native-coverage-engine none --known-bugs-file <known-bugs.json> --unresolved-failures-csv pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.csv --merge-unselected
```

Final report refresh:

```bash
python3 scripts/report.py <lib> --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage
```

Run manifest:

```bash
python3 scripts/write_run_manifest.py --lib <lib> --run-id <run_id> --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --out pipeline_runs/<lib>/<run_id>/run_manifest.json --seed 1337 --mutation-budget 8 --coverage-scope python --native-coverage unavailable_on_this_run
```

## Repair Workflow

Select low-coverage APIs:

```bash
python3 scripts/select_low_coverage_apis.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --threshold 10 --include-unavailable --max-apis 0 --out pipeline_runs/<lib>/<run_id>/repair/low_coverage_api_list.txt
```

Rerun selected low-coverage APIs:

```bash
LOW_COVERAGE_THRESHOLD=10 MUTATION_BUDGET=32 RUN_ID=<run_id> THESIS_API_LIST=pipeline_runs/<lib>/<run_id>/api_list.txt STAGE4_RESULTS_DIR=stage4/results/<lib>-thesis-all-coverage PYTHON_COV_SOURCE=<lib> PYTHON=python3 INIT_DIR=json2init/results/<lib> OK_CSV=json2init/results/<lib>/ok.csv COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 SEED=1337 ./scripts/rerun_low_coverage.sh <lib>
```

Select pipeline failures:

```bash
python3 scripts/select_failure_apis.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --include-audit --out pipeline_runs/<lib>/<run_id>/repair/failure_api_list.txt
```

Run the canonical repair entry point:

```bash
python3 scripts/repair.py <lib> --run-id <run_id> --dry-run --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --skip-stage4 --skip-report --force-validator --max-repair-iterations 1 --mutation-budget 8
```

## Saved Results

| Library | Accepted APIs | Frozen run list | Stage 2 valid | Stage 3 ready | Stage 4 evaluated | Base-valid APIs | Selected-function coverage | Valid programs | Candidate claims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JAX 0.10.0 | 477 | 379 | 477/477 | 476/477 | 476/477 accepted; 476/379 frozen | 476/477 accepted; 476/476 evaluated | 51.0843% | 1799 | 0 impl bugs, 0 doc mismatches |
| TensorFlow 2.21.0 | 922 | 705 | 904/922 | 705/922 | 705/922 accepted; 705/705 frozen | 212/922 accepted; 212/705 evaluated | 32.0496% over 74/705 source-resolved APIs | 973 | 0 impl bugs, 0 doc mismatches |
| PyTorch 2.11.0 | 593 | 323 | 593/593 | 323/593 | 323/593 accepted; 323/323 frozen | 323/593 accepted; 323/323 evaluated | 59.5745% | 1022 | 0 impl bugs, 0 doc mismatches |

Saved final reports:

- `pipeline_runs/jax/jax-thesis-all/final_report.md`
- `pipeline_runs/tensorflow/tensorflow-thesis-all/final_report.md`
- `pipeline_runs/torch/torch-thesis-all/final_report.md`

## Coverage And Bug Semantics

- The headline coverage metric is selected-function Python line coverage over APIs that reached Stage 4.
- API execution coverage is reported separately from source-line coverage.
- Package-level Python coverage is context only; whole-package denominators are much larger than the selected API subset.
- Native/kernel coverage is not claimed from binary wheels.
- Implementation bug candidates require isolated crash, hang, unexpected exception on intended-valid input, enabled NaN/Inf oracle failure, or CPU-vs-accelerator differential mismatch.
- Documentation mismatch candidates are kept separate from implementation bugs.
- Expected negative rejections, invalid generated inputs, permissive APIs, and pipeline failures are audit evidence rather than bug claims.

## Validation

Compile the core Python entry points:

```bash
python3 -m py_compile common/result_io.py common/model_config.py common/api_policy.py info2json/info2json.py json_validator/json_validator.py json2init/deepfuzz_common.py json2init/json2init.py stage4/stage4_worker.py stage4/stage4_coverage_runner.py stage4/known_bug_benchmarks.py scripts/select_api_subset.py scripts/read_coverage.py scripts/write_run_manifest.py scripts/repair.py scripts/report.py
```
