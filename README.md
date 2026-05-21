# DeepFuzz

- DeepFuzz turns deep-learning API documentation into executable fuzzing programs.
- Thesis goal: show an end-to-end, reproducible pipeline from docs to structured specs, seeds, isolated mutation runs, coverage, and strict bug/doc-mismatch triage.
- Final scope: JAX, TensorFlow, and PyTorch APIs that were accepted from docs and could be turned into Stage 3 executable seeds.
- Main metric: selected-function Python line coverage for each selected API.
- Secondary metrics: API execution coverage, generated valid programs, package-level Python coverage, and triage counts.
- Non-goal: whole-library native/kernel coverage from binary wheels.

## Setup

- Use Python 3.11 because it is the stable thesis environment for local JAX/TensorFlow/PyTorch wheels.

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt
```

- Pull local Ollama models used by the pipeline.

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen3-coder:30b
```

- Export model settings before generation, validation, or repair.

```bash
export DEEPFUZZ_MODEL_BACKEND=ollama
export OLLAMA_HOST=http://localhost:11434
export DEEPFUZZ_MODEL=qwen2.5:14b
export DEEPFUZZ_FAST_MODEL=qwen2.5-coder:7b
export DEEPFUZZ_REPAIR_MODEL=qwen2.5-coder:14b
export DEEPFUZZ_STRONG_REPAIR_MODEL=qwen3-coder:30b
```

## Basic Run

- Basic all-ready-API run for one library.
- Replace `<lib>` with `jax`, `tensorflow`, or `torch`.
- Expected outputs:
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

- JAX uses CPU mode for reproducibility on macOS.

```bash
JAX_PLATFORMS=cpu RUN_ID=jax-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/jax/jax-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/jax-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=jax MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/jax SPEC_DIR=info2json/results/jax VALIDATOR_OK_CSV=json_validator/results/jax/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh jax
```

- TensorFlow all-ready-API run.

```bash
RUN_ID=tensorflow-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/tensorflow/tensorflow-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/tensorflow-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=tensorflow MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/tensorflow SPEC_DIR=info2json/results/tensorflow VALIDATOR_OK_CSV=json_validator/results/tensorflow/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh tensorflow
```

- PyTorch all-ready-API run.

```bash
RUN_ID=torch-thesis-all SEED=1337 THESIS_API_LIST=pipeline_runs/torch/torch-thesis-all/api_list.txt STAGE4_RESULTS_DIR=stage4/results/torch-thesis-all-coverage COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=torch MUTATION_BUDGET=8 THESIS_LIMIT=0 INIT_DIR=json2init/results/torch SPEC_DIR=info2json/results/torch VALIDATOR_OK_CSV=json_validator/results/torch/ok.csv NATIVE_COVERAGE_STATUS=unavailable_on_this_run PYTHON=python3 ./scripts/run_thesis_final.sh torch
```

## Pipeline Commands

- Full templates below show every CLI argument for that command.
- Remove optional boolean flags and placeholder values unless that use case applies.

- API collection: import a library, walk public modules/classes, and write raw doc/signature entries.

```bash
python3 doc2info/collector.py --root <root> --out doc2info/<lib>_apis.jsonl --max-depth 5 --exclude-prefix <prefix> --no-descriptors --no-callables --no-classes --no-class-members
```

- API filter: keep APIs that are plausible one-call fuzz targets and write accepted/rejected evidence.

```bash
python3 doc2info/filter_accepted.py --in doc2info/<lib>_apis.jsonl --outdir doc2info/results --lib-name <lib> --max 0 --max-examples 50 --allow-receiver-apis --allow-wrapper-bridges
```

- Stage 1, docs to JSON specs: use the LLM to extract argument/constraint/return contracts.

```bash
python3 info2json/info2json.py --input doc2info/results/<lib>/accepted.csv --outdir info2json/results/<lib> --model "$DEEPFUZZ_MODEL" --host "$OLLAMA_HOST" --timeout 300 --limit 0 --retries 3 --sleep 0 --overwrite --combined-out info2json/results/<lib>/combined.json --num-predict 320 --num-ctx 2048 --temperature 0 --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt
```

- Stage 2, validate/repair specs: enforce schema and repair malformed or incomplete JSON.

```bash
python3 json_validator/json_validator.py --spec-dir info2json/results/<lib> --api-csv doc2info/results/<lib>/accepted.csv --state-dir json_validator/results/<lib> --primary-repair-model "$DEEPFUZZ_REPAIR_MODEL" --fallback-repair-model "$DEEPFUZZ_STRONG_REPAIR_MODEL" --fallback-after-round 3 --repair-host "$OLLAMA_HOST" --max-rounds 3 --repair-timeout 300 --repair-num-predict 700 --repair-num-ctx 4096 --repair-temperature 0 --only-apis pipeline_runs/<lib>/<run_id>/api_list.txt --force --external-errors-csv pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.csv --external-errors-jsonl pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.jsonl
```

- Stage 3, specs to executable init seeds: materialize base seeds and smoke-run them.

```bash
python3 json2init/json2init.py --spec-dir info2json/results/<lib> --outdir json2init/results/<lib> --ok-csv json_validator/results/<lib>/ok.csv --smoke-test --overwrite --reuse-existing --limit 0 --smoke-timeout-sec 30 --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt --non-strict-smoke
```

- Stage 4, isolated fuzzing and coverage: run each init file in a worker, mutate inputs, export reports.

```bash
python3 stage4/stage4_coverage_runner.py --init-dir json2init/results/<lib> --results-dir stage4/results/<lib>-thesis-all-coverage --ok-csv json2init/results/<lib>/ok.csv --mutation-budget 8 --seed 1337 --run-id <run_id> --only-api-list pipeline_runs/<lib>/<run_id>/api_list.txt --limit 0 --coverage-scope python --low-coverage-threshold 60 --worker-timeout-sec 120 --case-timeout-sec 30 --enable-python-coverage --python-cov-source <lib> --python-cov-omit <glob> --native-coverage-engine none --native-source-root <source-root> --native-build-dir <build-dir> --native-html --gcovr-executable gcovr --gcov-executable gcov --gcovr-filter <filter> --gcovr-exclude <exclude> --gcovr-extra-arg <arg> --gcovr-jobs 1 --known-bugs-file <known-bugs.json> --unresolved-failures-csv pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.csv --merge-unselected
```

- Stage 4 bug-finding oracle: on hosts with both CPU and an accelerator backend, enable the integrated differential device oracle. This is part of the same Stage 4 worker/bug-report path, not a separate extension. It runs valid base and valid-mutated calls on CPU plus the first available accelerator, compares output values with tolerance, preserves signed-zero differences, and records mismatches as `differential_mismatch` candidate implementation bugs.

```bash
ENABLE_DEVICE_ORACLE=1 EDGE_ORACLE_MUTATIONS=1 RUN_ID=<run_id> THESIS_API_LIST=pipeline_runs/<lib>/<run_id>/api_list.txt STAGE4_RESULTS_DIR=stage4/results/<lib>-thesis-all-coverage PYTHON_COV_SOURCE=<lib> PYTHON=python3 ./scripts/run_stage4.sh <lib> --enable-device-oracle --device-oracle-device cpu --device-oracle-device accelerator --edge-oracle-mutations --merge-unselected
```

- The edge oracle mutations add valid numerical probes such as NaN, infinities, signed zero, and large magnitudes for numeric tensor/scalar parameters. This is how the workflow can discover bugs similar to CPU/GPU cast divergence, float16 reduction divergence, and signed-zero activation divergence without hard-coding those APIs or expected outputs.

- Run manifest: record versions, model config, API-list hash, seed, budget, and coverage scope.

```bash
python3 scripts/write_run_manifest.py --lib <lib> --run-id <run_id> --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --out pipeline_runs/<lib>/<run_id>/run_manifest.json --seed 1337 --mutation-budget 8 --coverage-scope python --native-coverage unavailable_on_this_run
```

- Final report: combine Stage 1-4 artifacts into thesis tables and triage files.

```bash
python3 scripts/report.py <lib> --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage
```

- Read coverage: print the saved headline coverage numbers.

```bash
python3 scripts/read_coverage.py --stage4-results-dir stage4/results/<lib>-thesis-all-coverage
```

- Stage outputs:
  - Collection/filter: `doc2info/<lib>_apis.jsonl`, `doc2info/results/<lib>/accepted.csv`, rejected CSV/summary.
  - Stage 1: `info2json/results/<lib>/*.json`.
  - Stage 2: `json_validator/results/<lib>/ok.csv` and `errors.csv`.
  - Stage 3: `json2init/results/<lib>/*.init.json`, `ok.csv`, and `errors.csv`.
  - Stage 4: `results.csv`, `execution_summary.csv`, worker JSON, coverage reports, bug reports.
  - Final run: `pipeline_runs/<lib>/<run_id>/final_report.md`, JSON, triage, and repro files.

## Repair Commands

- Select low selected-function coverage APIs for targeted rerun.

```bash
python3 scripts/select_low_coverage_apis.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --threshold 10 --include-unavailable --max-apis 0 --out pipeline_runs/<lib>/<run_id>/repair/low_coverage_api_list.txt
```

- Rerun only low-coverage APIs and merge untouched prior evidence.

```bash
LOW_COVERAGE_THRESHOLD=10 MUTATION_BUDGET=32 RUN_ID=<run_id> THESIS_API_LIST=pipeline_runs/<lib>/<run_id>/api_list.txt STAGE4_RESULTS_DIR=stage4/results/<lib>-thesis-all-coverage PYTHON_COV_SOURCE=<lib> PYTHON=python3 INIT_DIR=json2init/results/<lib> OK_CSV=json2init/results/<lib>/ok.csv COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 SEED=1337 ./scripts/rerun_low_coverage.sh <lib>
```

- Select pipeline failures for targeted rerun.

```bash
python3 scripts/select_failure_apis.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --api-list pipeline_runs/<lib>/<run_id>/api_list.txt --include-audit --out pipeline_runs/<lib>/<run_id>/repair/failure_api_list.txt
```

- Rerun selected failures while preserving unselected rows.

```bash
RUN_ID=<run_id> SEED=1337 MUTATION_BUDGET=8 COVERAGE_SCOPE=python ENABLE_PYTHON_COVERAGE=1 PYTHON_COV_SOURCE=<lib> STAGE4_RESULTS_DIR=stage4/results/<lib>-thesis-all-coverage STAGE4_MERGE_UNSELECTED=1 PYTHON=python3 ./scripts/run_stage4.sh <lib> --only-api-list pipeline_runs/<lib>/<run_id>/repair/failure_api_list.txt --merge-unselected
```

- Refresh reports without rerunning APIs.

```bash
python3 scripts/refresh_stage4_reports.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --known-bugs-file <known-bugs.json> --unresolved-failures-csv pipeline_runs/<lib>/<run_id>/triage/pipeline_failures.csv
```

- Snapshot before or after repair.

```bash
python3 scripts/snapshot_run.py --lib <lib> --run-id <run_id> --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --label before-lowcov-b32
```

- Canonical repair entry point.

```bash
python3 scripts/repair.py <lib> --run-id <run_id> --dry-run --stage4-results-dir stage4/results/<lib>-thesis-all-coverage --skip-stage4 --skip-report --force-validator --max-repair-iterations 1 --mutation-budget 8
```

- Optional JAX known-bug calibration.

```bash
python3 stage4/known_bug_benchmarks.py --outdir stage4/results/jax-known-bug-benchmarks --benchmark B1
```

## Option Use Cases

- `--overwrite`: optional; replace existing generated outputs when prompts, schema, or extraction settings changed.
- `--override`: not a current DeepFuzz flag; use `--overwrite` when you want replacement behavior.
- `--reuse-existing`: optional; keep already-ready Stage 3 init files to avoid regenerating stable seeds.
- `--only-api-list`: optional; freeze the thesis denominator or rerun a targeted subset.
- `--limit 0`: no cap; use `--limit N` for quick smoke/debug runs only.
- `--all`: write every ready Stage 3 API into the frozen list.
- `--public-only`: optional ablation; excludes internal APIs before final evaluation.
- `--force`: optional; revalidate APIs already marked valid.
- `--merge-unselected`: important for repair; merges old rows for APIs outside the rerun subset.
- `--coverage-scope`: choose `none`, `api_only`, `python`, `native`, `per_api`, `campaign`, or `both`; thesis headline uses `python`.
- `--enable-python-coverage` and `--python-cov-source`: enable coverage.py for the selected package.
- `--enable-device-oracle`: compare valid executions across CPU and accelerator devices when both are available.
- `--edge-oracle-mutations`: include NaN/Inf/signed-zero/large-value probes in valid mutation generation; automatically enabled by `--enable-device-oracle`.
- `--native-*` and `--gcovr-*`: only meaningful with an instrumented native source build.
- `--allow-receiver-apis` and `--allow-wrapper-bridges`: filter relaxations; not used for the conservative final claim.
- `--repair-model`: backward-compatible alias for `--primary-repair-model`; prefer the explicit primary/fallback names.

## Current Results

| Library | Accepted APIs | Frozen run list | Stage 2 valid | Stage 3 ready | Stage 4 evaluated | Base-valid APIs | Selected-function coverage | Valid programs | Candidate claims | Left |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JAX 0.10.0 | 477 | 379 | 477/477 | 476/477 | 476/477 accepted; 476/379 frozen | 476/477 accepted; 476/476 evaluated | 51.0843% | 1799 | 0 impl bugs, 0 doc mismatches | Regenerate public-only frozen list: 97 ready APIs are outside the frozen list; 1 CPU platform exclusion |
| TensorFlow 2.21.0 | 922 | 705 | 904/922 | 705/922 | 705/922 accepted; 705/705 frozen | 212/922 accepted; 212/705 evaluated | 32.0496% over 74/705 source-resolved APIs | 973 | 0 impl bugs, 0 doc mismatches | 199 seed/init failures and 1185 pipeline errors to repair |
| PyTorch 2.11.0 | 593 | 323 | 593/593 | 323/593 | 323/593 accepted; 323/323 frozen | 323/593 accepted; 323/323 evaluated | 59.5745% | 1022 | 0 impl bugs, 0 doc mismatches | 251 seed/init failures, 19 platform/runtime exclusions, 251 pipeline errors |

- Current saved reports:
  - `pipeline_runs/jax/jax-thesis-all/final_report.md`
  - `pipeline_runs/tensorflow/tensorflow-thesis-all/final_report.md`
  - `pipeline_runs/torch/torch-thesis-all/final_report.md`
- JAX is clean for the evaluated public subset, but the current frozen list was stale/mixed-scope; rerun it from a regenerated public-only API list before using the JAX row as final.
- TensorFlow/PyTorch still prove the pipeline generalizes, but their base-valid counts and source-resolved coverage availability show remaining framework-specific seed and runtime work.
- A zero-bug CPU-only run is still a valid conservative result because the oracle rejects weak claims and separates pipeline failures, invalid mutations, expected negative rejections, permissive APIs, and documentation mismatches. For the thesis bug-finding story, rerun Stage 4 with `--enable-device-oracle` on a CPU+GPU host and file only replay-confirmed `differential_mismatch`, crash, timeout, or documented-contract violations.

## Bug-Finding Oracles

- DeepFuzz follows the VISTAFUZZ-style document-guided idea: extract standardized API information and constraints from documentation, then use those constraints to generate systematic programs. The DL-library extension is that valid executions also run through oracles that can expose silent numerical bugs.
- Implementation bug candidates include native crashes, hangs, unexpected exceptions on intended-valid inputs, NaN/Inf outputs when that oracle is enabled, and CPU-vs-accelerator `differential_mismatch` results.
- Documentation mismatch candidates are kept separate: a report belongs there only when the generated input is consistent with the documented contract but the implementation rejects it, or when documentation declares behavior that the API demonstrably does not follow.
- Negative mutations that are accepted by permissive APIs stay in `negative_accepted_not_bug.csv`; they are audit evidence, not bug claims.
- Worker-level timeouts after a successful API bundle are treated as pipeline/coverage failures, not library bugs. Only timeouts or crashes attributable to an isolated API case become candidate implementation bugs.

## Coverage Decision

- Chosen method: run coverage.py inside isolated Stage 4 workers, resolve each API's Python source span with `inspect`, and aggregate selected-function line coverage.
- Why this is correct:
  - It measures the function/wrapper lines reached by generated programs.
  - It avoids pretending binary-wheel native kernels have source coverage.
  - It keeps API execution coverage separate from source-line coverage.
  - It keeps package-level Python coverage as context because whole-package denominators are huge and misleading for selected APIs.
- Coverage alternatives considered:
  - API execution only: too weak; says the API ran, not how much code was touched.
  - Package-level Python coverage only: too diluted by unselected framework code.
  - Native coverage: useful but requires instrumented source builds, so it is appendix-only unless actually run.
  - Campaign-level coverage only: easier, but weaker for per-API repair because it cannot cleanly identify low-coverage APIs.

## Major Decisions

- Pipeline denominator: use accepted documented APIs, then report Stage 2, Stage 3, and Stage 4 funnels separately.
- Final Stage 4 input: use `json2init/results/<lib>/ok.csv` and `*.init.json`, not raw documentation rows, because Stage 4 needs executable seeds.
- All-ready APIs over manual subsets: manual subsets were useful during development, but final results should use every API that reached Stage 3 readiness.
- Targeted repair over full reruns: rerun only failures/low-coverage APIs and preserve old evidence with `--merge-unselected`.
- Strict oracle over optimistic bug counting: only replay-confirmed implementation bugs or documentation mismatches enter `bug_report.csv`.
- Integrated CPU/accelerator differential oracle: device comparison is part of Stage 4 and `bug_report.csv`, so finding bugs and reporting coverage share the same artifacts and replay path.
- Local Ollama over hosted APIs: reproducible, local, and low cost for repeated thesis runs.
- Model split:
  - `qwen2.5:14b`: main doc-to-spec generation.
  - `qwen2.5-coder:7b`: faster lightweight coding/spec help.
  - `qwen2.5-coder:14b`: normal repair.
  - `qwen3-coder:30b`: stronger but slower; reserve for small unresolved repair batches.
- Model effect: stronger models can improve hard repairs, but full-library generation becomes slower and harder to rerun; deterministic temperature `0` keeps runs stable.
- Docker decision: Docker files were useful for old smoke/native experiments, but the final thesis run uses local macOS Python wheels, Ollama, and coverage.py; Docker adds environment friction without improving the main Python coverage claim.
- Native/source-build decision: do not claim native coverage unless a real instrumented source build is run.

## Thesis Status

- Achieved:
  - Documentation collection and filtering.
  - LLM-generated structured API contracts.
  - Schema validation and repair.
  - Dependency-aware seed materialization.
  - Isolated mutation execution.
  - Per-API selected-function coverage.
  - Strict bug/doc-mismatch triage.
  - Final reports for JAX, TensorFlow, and PyTorch.
- Why it is good overall:
  - It is honest about denominators and exclusions.
  - It does not inflate bugs from invalid mutations or pipeline errors.
  - It produces reproducible artifacts and commands.
  - It supports targeted repair instead of throwing away previous evidence.
- What is left:
  - Repair TensorFlow/PyTorch pipeline failures.
  - Finish low-coverage reruns for selected APIs.
  - Optionally add a Docker/native coverage appendix if time allows.
  - Write the thesis discussion around the saved final reports.
- Thesis completion argument:
  - The end-to-end system works.
  - The strongest result, JAX, demonstrates the full method clearly.
  - TensorFlow and PyTorch add cross-framework evidence and show realistic limitations.
  - Remaining work is improvement and polishing, not inventing the pipeline from scratch.

## Validation

- Compile changed Python files.

```bash
python3 -m py_compile common/result_io.py common/model_config.py common/api_policy.py info2json/info2json.py json_validator/json_validator.py json2init/deepfuzz_common.py json2init/json2init.py stage4/stage4_worker.py stage4/stage4_coverage_runner.py stage4/known_bug_benchmarks.py scripts/select_api_subset.py scripts/read_coverage.py scripts/write_run_manifest.py scripts/repair.py scripts/report.py
```
