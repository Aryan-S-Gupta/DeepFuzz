# DeepFuzz Poster Demo Script

## Source Run Used

The poster uses the committed final thesis artefacts:

- `pipeline_runs/jax/jax-thesis-all`
- `pipeline_runs/tensorflow/tensorflow-thesis-all`
- `pipeline_runs/torch/torch-thesis-all`
- `stage4/results/jax-thesis-all-coverage`
- `stage4/results/tensorflow-thesis-all-coverage`
- `stage4/results/torch-thesis-all-coverage`

The poster numbers are taken from `final_report.json`, `coverage_report.json`, `bug_report.csv`, `run_manifest.json`, generated tests, and seed JSON files. The old project details PDF is treated as historical background only.

## 30-Second Elevator Pitch

DeepFuzz converts deep-learning API documentation into executable fuzzing artefacts. It extracts structured API specs from documentation, validates and repairs them, generates deterministic seed inputs, runs isolated mutation-based executions, and produces replayable pytest files, coverage reports, and triage files. The final thesis runs cover accepted JAX, TensorFlow, and PyTorch APIs, using selected-function Python line coverage as the main metric because whole-library native coverage from binary wheels is misleading for this scope.

## 3-5 Minute Poster-Guided Pitch

Start with the problem. Deep-learning libraries are large, fast moving, and difficult to test manually. Their documentation describes intended behaviour, but documentation is not directly executable and can be incomplete or ambiguous.

Move to the aim. The thesis goal was to build an end-to-end, reproducible pipeline that turns documentation into executable fuzz tests and evidence across JAX, TensorFlow, and PyTorch.

Walk through the central pipeline. Stage 0 selects plausible one-call public APIs. Stage 1 turns documentation into structured API specs. Stage 2 validates JSON against schemas and runtime evidence, with local Ollama repair where needed. Stage 3 creates deterministic `.init.json` seeds and smoke-tests them. Stage 4 runs isolated fuzz cases, records failures and timeouts, measures selected-function coverage, and generates pytest tests for replay.

Point to the architecture box. The system combines documentation, runtime introspection, local LLM generation and repair, validators, the seed generator, isolated workers, coverage instrumentation, and triage reports. Device-oracle and edge numeric mutation support were enabled in the final Stage 4 runs.

Summarise the results table. The final runs accepted 1,992 APIs, generated 1,407 frozen pytest tests, and produced 12,383 valid generated programs. API execution coverage was 100.0% for JAX, 98.8652% for TensorFlow, and 100.0% for PyTorch. Selected-function Python line coverage was 52.3032% for JAX, 31.2415% for TensorFlow, and 59.5745% for PyTorch.

Explain the triage language carefully. The poster reports candidate implementation-bug records and device-oracle mismatch records, not confirmed upstream bugs. Negative mutations are expected rejection tests, and they are separated from documentation mismatches and pipeline failures.

Close with significance and limits. DeepFuzz is useful because it creates an auditable chain from docs to tests to reports, making generated fuzzing evidence reproducible and inspectable. The main limitation is scope: the headline metric is selected-function Python coverage, while whole-library native/kernel coverage from binary wheels is outside the final thesis scope and recorded as unavailable on these runs.

## Live Demo Path

Run the commands from `/Users/aryansg/Desktop/DeepFuzz`.

1. Show the final coverage summary for the largest candidate-bug run:

```bash
python3 scripts/read_coverage.py --stage4-results-dir stage4/results/torch-thesis-all-coverage
```

2. Show one deterministic seed generated from documentation:

```bash
sed -n '1,160p' json2init/results/jax/jax.numpy.sin.init.json
```

3. Show the generated replay test for the same API:

```bash
sed -n '1,140p' pipeline_runs/jax/jax-thesis-all/generated_tests/test_jax_numpy_sin.py
```

4. Replay the fast generated pytest:

```bash
python3 -m pytest pipeline_runs/jax/jax-thesis-all/generated_tests/test_jax_numpy_sin.py -q
```

5. Show device-oracle candidate examples from the PyTorch triage report:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("stage4/results/torch-thesis-all-coverage/bug_report.csv")
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows[:8]:
    print(row["api"], row["category"], row["classification"], row["exit_code"])
PY
```

6. If there is time, connect the files:

```bash
ls pipeline_runs/jax/jax-thesis-all/generated_tests | head
ls stage4/results/torch-thesis-all-coverage | sort
```

## Backup Recorded-Demo Plan

If live execution is slow or the local environment is busy, use existing final artefacts instead of re-running fuzzing:

- Open `stage4/results/torch-thesis-all-coverage/coverage_report.md`.
- Open `stage4/results/torch-thesis-all-coverage/bug_report.csv`.
- Open `json2init/results/jax/jax.numpy.sin.init.json`.
- Open `pipeline_runs/jax/jax-thesis-all/generated_tests/test_jax_numpy_sin.py`.
- Explain that Stage 4 full runs are intentionally long because they isolate processes, collect coverage, run mutations, and record triage evidence.

## Likely Examiner Questions And Answers

**Why selected-function coverage instead of whole-library coverage?**  
Whole-library package coverage is dominated by import paths, wrappers, and binary/native backend code. The thesis measures Python lines inside the selected API functions because that reflects whether the generated tests exercised the APIs under study.

**Are the candidate bugs confirmed upstream bugs?**  
No. They are triaged candidate records. DeepFuzz separates candidate implementation bugs, device-oracle mismatches, documentation mismatches, expected negative rejections, and pipeline failures so the evidence can be reviewed before any upstream report.

**Why use documentation if documentation can be wrong?**  
That is the core research idea: documentation provides intended behaviour, but DeepFuzz grounds it with schemas, runtime introspection, smoke tests, isolated execution, and triage. A mismatch becomes evidence, not an unexamined assumption.

**What role did local LLMs play?**  
Local Ollama models were used for documentation-to-structure extraction and repair of malformed or incomplete specs. The pipeline still requires validation before a spec becomes an executable seed.

**What makes this complete rather than just a prototype?**  
The repo contains implemented stages, run manifests, final reports, generated seeds, generated pytest files, coverage reports, execution summaries, and triage CSV/JSON outputs for the final thesis runs.

**Why are PyTorch mismatch counts much larger?**  
The PyTorch final run produced many device-oracle mismatch records. They are reported conservatively as candidates and need manual review; the important engineering point is that they are isolated from expected negative tests and pipeline errors.

**Can this support another library?**  
The architecture is library-agnostic in principle, but a new library still needs API selection, documentation collection, runtime import handling, seed materialisation support, and final validation.
