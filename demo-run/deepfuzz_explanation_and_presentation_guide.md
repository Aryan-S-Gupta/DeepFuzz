# DeepFuzz Explanation And Presentation Guide

This guide is written for you, not for a paper reviewer. Use it to explain the project from scratch, defend the design choices, walk through the poster, and run the live demo.

## 1. What You Are Doing

You built DeepFuzz, an end-to-end pipeline that turns deep-learning API documentation into executable fuzz tests and evidence. The point is not just "LLM writes tests". The real point is: documentation is treated as a source of intended behaviour, but every claim is forced through validation, seed generation, isolated execution, coverage measurement, and triage.

In one sentence:

> DeepFuzz converts API documentation into reproducible fuzzing artefacts: JSON specs, executable seeds, generated pytest tests, coverage reports, and bug/triage files.

The important story is that this is a software-engineering pipeline, not a one-off prompt. Each stage creates files that can be inspected, rerun, or challenged.

Talk bullets:

- "I am not asking an LLM to magically find bugs."
- "I am building a controlled pipeline from docs to executable evidence."
- "Every stage leaves artefacts: specs, seeds, tests, reports, and triage CSV/JSON."
- "The final claim is reproducibility and auditability."

## 2. Motivation: Why This Space Matters

Deep-learning libraries are huge and change constantly. JAX, TensorFlow, and PyTorch expose many APIs across tensors, math, neural-network layers, masking, shapes, devices, dtypes, random/stateful operations, and lower-level runtime features. Manual tests cannot realistically cover all combinations of argument types, shapes, values, dtypes, and devices.

The key observation is that API documentation already contains valuable test information:

- parameter names
- types
- default values
- valid ranges or shapes
- examples
- error behaviour
- return descriptions

But documentation is prose. It is written for humans, not machines. DeepFuzz explores whether that prose can be turned into executable fuzzing evidence.

Why this was necessary:

- Docs often say what should be valid, but the implementation decides what actually runs.
- Examples usually test only happy paths, not edge cases.
- ML libraries run across CPU/GPU/accelerator backends, where numerical differences can appear.
- Version changes can silently alter behaviour.
- Traditional fuzzers need input grammars or harnesses; API-level ML fuzzing needs valid tensors, shapes, dtypes, and constraints.

Real-world applications:

- Generate regression tests for library upgrades.
- Detect documentation/API mismatches before users hit them.
- Compare CPU vs GPU/accelerator behaviour.
- Produce reproducible bug reports for maintainers.
- Audit API coverage for large libraries.
- Help developers test wrapper libraries that call JAX/TensorFlow/PyTorch.
- Build CI smoke tests for selected APIs.

Talk bullets:

- "The hard part is not calling an API once. The hard part is producing valid, meaningful, reproducible calls at scale."
- "Docs describe intent; runtime validates reality."
- "DeepFuzz bridges that gap."

## 3. Important Keywords

**API**: A callable function or operation exposed by a library, such as `jax.numpy.sin`, `tensorflow.nn.relu`, or `torch.masked.sum`.

**Deep-learning API**: An API that usually works on tensors, arrays, devices, dtypes, shapes, masks, reductions, activations, or numerical values.

**Fuzzing**: Automated testing that generates many inputs to explore program behaviour. Here, the inputs are API calls and argument values.

**Documentation-driven fuzzing**: Fuzzing where documentation is used to infer valid inputs, invalid inputs, constraints, and expected behaviour.

**Spec**: A structured JSON description of an API. It contains the API name, module path, parameters, types, defaults, constraints, and output description.

**Seed**: A deterministic base input generated from a spec. In DeepFuzz it is saved as a `.init.json` file.

**Base-valid execution**: The seed input runs successfully. This is the first proof that the spec can become executable.

**Mutation**: A systematic change to a seed input. A valid mutation should still be accepted. A negative mutation is intentionally invalid and should usually be rejected.

**Valid program**: A base execution or valid mutation that successfully runs the API.

**Oracle**: A rule that decides whether behaviour is suspicious. DeepFuzz uses oracles such as crash/timeout detection, expected rejection checks, and CPU-vs-accelerator comparison.

**Device oracle**: Runs the same valid input on CPU and an accelerator when available, then checks whether outputs differ beyond tolerance.

**Edge numeric mutation**: A mutation using values like NaN, infinity, signed zero, or very large magnitudes.

**Coverage**: A measure of what code was exercised. DeepFuzz reports API execution coverage and, in final runs, selected-function Python line coverage.

**Selected-function coverage**: Coverage only over the Python source lines of the selected API functions. This is the headline metric because whole-library coverage is misleading for giant ML libraries.

**Package-level Python coverage**: Coverage over the whole installed Python package. This number is usually tiny because importing and running a few APIs touches only a small fraction of the package.

**Native/kernel coverage**: Coverage inside compiled backend kernels. This needs an instrumented source build and is not available from normal binary wheels.

**Candidate bug**: A triage record that looks like a possible implementation bug. It is not automatically a confirmed upstream bug.

**Documentation mismatch**: The docs imply an input should be valid, but runtime rejects it, or the implementation behaves differently from documented behaviour.

**Pipeline failure**: A failure in the generator, seed materializer, environment, coverage tooling, or harness. These are not library bugs.

**Frozen API list**: The fixed list of APIs used as the denominator for a run. "Frozen" means the list does not silently change during the experiment.

**Generated pytest**: A replayable Python test file that calls DeepFuzz on a seed with mutation budget 0. It proves the seed remains executable.

## 4. Why JAX, TensorFlow, And PyTorch

You chose these libraries because they are the dominant Python deep-learning ecosystems and they stress different parts of the pipeline.

JAX:

- NumPy-like APIs, transformations, functional style, and accelerator-aware execution.
- Good for array/math and selected-function coverage.

TensorFlow:

- Large API surface, Keras integration, graph/eager behaviour, many wrappers.
- Good stress test for documentation extraction and runtime complexity.

PyTorch:

- Python-first style, tensor ops, masked APIs, device backends, many C++/native kernels.
- Good stress test for device oracle and candidate mismatch triage.

Why not every library:

- The thesis needed a controlled scope.
- These three already represent a broad, meaningful DL ecosystem.
- Adding more libraries would mostly test adapter effort, not the core research idea.

How it generalises:

To support another library, you write an adapter/configuration layer for:

- collecting public APIs
- filtering safe one-call targets
- importing APIs by name
- materialising library-specific tensors/dtypes/devices
- moving inputs across devices
- choosing coverage source packages
- handling known environment exclusions

The core pipeline stays the same: docs -> specs -> validation -> seeds -> fuzz runs -> coverage/triage -> generated tests.

Talk bullets:

- "The pipeline is library-agnostic at the architecture level."
- "The adapter work is where library-specific tensor/device/dtype behaviour lives."
- "JAX, TensorFlow, and PyTorch were enough to show that this is not a single-library trick."

## 5. API Types You Are Testing

The project sees several API types:

- Elementwise numeric APIs: `sin`, `sinh`, `exp`, `relu`.
- Activation functions: `jax.nn.relu`, `tensorflow.nn.relu`.
- Reductions: `sum`, `mean`, `argmax`, `argmin`.
- Masked tensor operations: `torch.masked.sum`, `torch.masked.softmax`.
- Shape/dtype APIs: APIs that manipulate shape or dtype.
- Linear algebra APIs: norms, decompositions, solves.
- Runtime/device APIs: memory or device-related functions.
- I/O/image/string APIs: more complex and often harder to seed safely.

The live demo uses:

- `jax.numpy.sin`: elementwise numerical function.
- `jax.nn.relu`: neural-network activation.
- `tensorflow.nn.relu`: TensorFlow activation.
- `tensorflow.keras.ops.softmax`: probability/normalisation operation.
- `torch.masked.sum`: masked reduction.
- `torch.masked.softmax`: masked probability/normalisation operation.

These are useful demo APIs because they are familiar, fast, tensor-oriented, and easy to explain.

## 6. Why Filter APIs Instead Of Testing Everything

DeepFuzz filters because not every public API is a good one-call fuzz target.

Filtering removes or de-prioritises APIs that:

- require files, network, global state, sessions, or external services
- mutate global runtime state
- require complex multi-step object setup
- are class constructors rather than simple call targets
- are private/internal implementation helpers
- are environment-specific, such as GPU-only APIs on a CPU machine

This is not weakness. It is experimental discipline. Without filtering, the run would be dominated by noise: import errors, missing devices, invalid harnesses, and false positives.

Talk bullets:

- "The aim was not to call every symbol blindly."
- "The aim was to build a reliable documentation-to-execution pipeline."
- "Filtering keeps the denominator meaningful."

## 7. Methodology And Design Choices

### Stage 0: API Selection

DeepFuzz collects public APIs and filters them to plausible one-call fuzz targets.

Why:

- The pipeline needs a stable denominator.
- Fuzzing unsafe or stateful APIs would create noise.
- A fixed list makes results reproducible.

Artefacts:

- `accepted.csv`
- `rejected.csv`
- `api_list.txt`

Presentation example:

- "This is where I decide what is in scope for the experiment."
- "The frozen API list is the contract for the run."

### Stage 1: Documentation To Spec

Documentation is converted into structured JSON specs.

Why:

- Natural-language docs are not executable.
- JSON is machine-readable and auditable.
- Later stages can validate and repair JSON deterministically.

Artefacts:

- `info2json/results/<lib>/<api>.json`

What is inside:

- API name
- module path
- parameters
- parameter types
- defaults
- constraints
- output information

### Stage 2: Validation And Repair

Specs are validated against schema and runtime evidence. Broken specs can be repaired with local Ollama LLMs.

Why:

- LLM output can be malformed or unsupported.
- Docs can be ambiguous.
- Runtime signatures expose fake parameters or missing parameters.

Artefacts:

- `json_validator/results/<lib>/ok.csv`
- `json_validator/results/<lib>/errors.csv`

Design choice:

- Do not trust the LLM blindly.
- Make validation a gate before execution.

### Stage 3: Executable Seed Generation

Validated specs are turned into `.init.json` seeds.

Why:

- Fuzzing needs concrete values.
- A seed proves the documentation-derived spec can be executed.
- Deterministic seeds allow replay.

Artefacts:

- `json2init/results/<lib>/<api>.init.json`
- `json2init/results/<lib>/ok.csv`
- `json2init/results/<lib>/errors.csv`

Important field:

- `ready_for_stage4: true` means the seed is eligible for fuzzing.

### Stage 4: Isolated Fuzzing, Coverage, And Triage

Each API seed is run in an isolated worker. DeepFuzz creates valid mutations and negative mutations, records execution results, exports coverage, and writes bug reports.

Why isolation matters:

- A crash or abort in one API should not kill the whole campaign.
- Timeouts are contained.
- Evidence is stored per API.

Artefacts:

- `results.csv`
- `execution_summary.csv`
- `failure.csv`
- `coverage_report.json`
- `coverage_report.md`
- `bug_report.csv`
- `bug_report.json`
- `execution_results/<api>.results.json`

Design choice:

- Negative mutations are expected rejection tests.
- They are not automatically bugs.

## 8. What The Poster Sections Mean

Title:

- States the thesis: documentation-driven fuzz testing for deep-learning APIs.

Motivation / Problem:

- Explains why this is needed: big APIs, fast-changing libraries, docs not executable.

Research Aim:

- Convert docs into deterministic fuzz tests and evidence.

Pipeline Diagram:

- The central "how it works" story:
  docs -> specs -> validated JSON -> seeds -> isolated fuzz runs -> reports -> generated tests.

Implementation Architecture:

- Shows the engineering components: docs, runtime introspection, local LLMs, validators, seed generator, isolated runner, coverage/triage outputs.

Results Table:

- Shows final thesis numbers for JAX, TensorFlow, and PyTorch.

Coverage Chart:

- Shows selected-function Python line coverage, not whole-library native coverage.

Readiness Chart:

- Shows accepted APIs narrowing into generated tests and base-valid APIs.

Triage Chart:

- Shows why bug claims are conservative. Expected negative rejections, invalid mutations, non-bug negative acceptances, candidate bugs, and device mismatches are separated.

Demo Artefacts Box:

- Tells the examiner what you can show live: seed JSON, generated pytest, coverage report, bug report, and one API travelling through the pipeline.

Contribution Box:

- Highlights the novelty: docs-first, runtime-grounded, multi-stage validation, reproducible manifests, selected-function coverage, strict triage.

Scope Boundaries:

- Makes the limitation explicit: no fake native/kernel coverage from binary wheels.

## 9. Final Poster Results Explained

Final table:

| Library | Accepted APIs | Frozen/generated tests | Base-valid APIs | API exec. coverage | Selected-function coverage | Valid programs | Candidate implementation bugs | Device mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JAX 0.10.0 | 477 | 379 | 476/476 | 100.0% | 52.3032% | 3,421 | 2 | 0 |
| TensorFlow 2.21.0 | 922 | 705 | 697/705 | 98.8652% | 31.2415% | 6,866 | 3 | 0 |
| PyTorch 2.11.0 | 593 | 323 | 323/323 | 100.0% | 59.5745% | 2,096 | 185 | 1,209 |

What the numbers mean:

- Accepted APIs: documentation candidates accepted into the thesis scope.
- Frozen/generated tests: APIs fixed into the run list and exported as pytest files.
- Base-valid APIs: APIs whose deterministic seed ran successfully.
- API execution coverage: base-valid APIs divided by evaluated APIs.
- Selected-function coverage: Python source lines inside selected API functions reached by generated executions.
- Valid programs: base-valid runs plus successful valid mutations.
- Candidate implementation bugs: possible bugs under DeepFuzz triage, not confirmed upstream bugs.
- Device mismatches: CPU-vs-accelerator output differences detected by the device oracle.

Good results:

- JAX and PyTorch reached 100% API execution coverage on evaluated APIs.
- TensorFlow reached 98.8652%, which is still high for a large and complex library.
- The pipeline generated 1,407 replayable pytest files.
- The final runs produced 12,383 valid programs.
- The results are auditable through JSON/CSV/pytest artefacts.

Bad or limited results:

- TensorFlow selected-function coverage is lower because TensorFlow has wrappers, graph/eager layers, and source spans that can be hard to attribute cleanly.
- Package-level Python coverage is tiny because the selected APIs touch only a small fraction of huge packages.
- Native/kernel coverage is unavailable because binary wheels are not compiled with the needed instrumentation.
- PyTorch has many device-oracle mismatch records, but these are candidates requiring manual confirmation.
- Candidate bugs are not automatically reportable upstream until reduced and reproduced.

How to say this:

- "The good result is that the pipeline works end-to-end at scale."
- "The honest limitation is that coverage and bug status must be interpreted carefully."
- "I deliberately avoid claiming confirmed upstream bugs without manual confirmation."

## 10. Are The Bugs Real?

Short answer:

> They are candidate bugs, not confirmed bugs.

DeepFuzz records a candidate implementation bug when valid generated inputs cause suspicious behaviour such as:

- crash or abort
- timeout or hang
- unexpected exception on intended-valid input
- differential mismatch between CPU and accelerator outputs
- NaN/Inf output when that oracle is enabled and the behaviour is suspicious

DeepFuzz does not count these as bugs:

- negative mutations rejected by normal validation
- invalid valid mutations generated by the pipeline
- missing dependency or environment issue
- seed materialisation failure
- pipeline timeout not attributable to a specific API
- documentation ambiguity without runtime proof

Can you report them to library developers with confidence?

Not immediately from the table alone. For upstream reporting, you should:

- reproduce the candidate with the saved command or result JSON
- reduce it to a minimal script without the DeepFuzz harness
- test on latest stable and, if possible, nightly/dev version
- verify the input is valid according to docs and runtime type expectations
- check whether behaviour is documented, known, or platform-specific
- include exact versions, device details, input values, expected vs actual output
- report as "possible bug" or "device inconsistency" until maintainers confirm

The PyTorch device mismatch count is especially important to phrase conservatively. It means the oracle observed many CPU/accelerator output differences in that run. It does not mean 1,209 distinct confirmed PyTorch bugs.

Talk bullets:

- "The pipeline produces candidates, not legal verdicts."
- "The value is that candidates are reproducible and separated from expected failures."
- "The next step for upstream reporting is minimisation and confirmation."

## 11. CPU vs GPU / Accelerator Difference

The device oracle tries to run the same valid input on CPU and an accelerator. It then compares outputs with a numerical tolerance.

Why this matters:

- ML libraries often dispatch to different kernels on different devices.
- Floating-point behaviour can differ across hardware.
- Some bugs are silent: no crash, but wrong or inconsistent values.

What happens if there is no accelerator:

- The oracle records that it was enabled but unavailable or unable to compare two devices.
- This is not a failure. It is honest environment reporting.

How to explain:

- "The live demo may show zero device mismatches because my current machine may not expose a comparable accelerator for that backend."
- "The final PyTorch run did enable the device oracle and found candidate mismatch records."
- "A mismatch is a lead for manual investigation, not automatic proof."

## 12. Demo Run Results

The live demo is saved in:

`demo-run/results/latest`

Command:

```bash
.venv/bin/python demo-run/run_demo.py --api-file demo-run/api_list.txt --out demo-run/results/latest --force --mutation-budget 2 --enable-device-oracle
```

Generated pytest verification:

```bash
.venv/bin/python -m pytest demo-run/results/latest/jax/generated_tests demo-run/results/latest/tensorflow/generated_tests demo-run/results/latest/torch/generated_tests -q
```

Observed verification:

```text
6 passed in 14.20s
```

Demo summary:

| Library | APIs | Stage 2 valid | Stage 3 ready | Base-valid | API exec. cov. | Selected-function cov. | Valid programs | Candidate bugs | Device mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jax | 2 | 2 | 2 | 2/2 | 100.0% | unavailable | 6 | 0 | 0 |
| tensorflow | 2 | 2 | 2 | 2/2 | 100.0% | unavailable | 6 | 0 | 0 |
| torch | 2 | 2 | 2 | 2/2 | 100.0% | unavailable | 8 | 0 | 0 |

Why selected-function coverage is unavailable in the live demo:

- The live run uses `--coverage-scope api_only` for speed.
- This avoids slow TensorFlow/PyTorch coverage export during the presentation.
- The final poster uses the full saved thesis runs for selected-function coverage.

This is a good demo result because:

- all six APIs passed Stage 2 validation
- all six produced Stage 3 executable seeds
- all six ran base-valid in Stage 4
- generated pytest replay tests passed
- valid mutation programs were generated
- no candidate bugs were claimed from this small run
- reports and triage files were produced

This is not a bug-finding showcase by itself. It is an end-to-end functionality showcase. The full thesis runs are the evidence for scale and candidate triage.

## 13. Demo Artefact Structure

Top-level:

- `demo-run/api_list.txt`: default six APIs.
- `demo-run/run_demo.py`: reusable mini-pipeline.
- `demo-run/results/latest/demo_report.md`: summary table.
- `demo-run/results/latest/commands.log`: exact commands.
- `demo-run/results/latest/all_apis.txt`: APIs used in the run.

Per library:

- `<lib>/stage0_selection/api_list.txt`: frozen API list for that mini-run.
- `<lib>/stage0_selection/accepted_subset.csv`: documentation rows for those APIs.
- `<lib>/stage1_specs/*.json`: structured documentation-derived API specs.
- `<lib>/stage2_validator/ok.csv`: specs that passed validation.
- `<lib>/stage2_validator/errors.csv`: validation failures, if any.
- `<lib>/stage3_init/*.init.json`: executable seed objects.
- `<lib>/stage3_init/ok.csv`: seeds ready for Stage 4.
- `<lib>/generated_tests/test_*.py`: replayable pytest tests.
- `<lib>/run_manifest.json`: reproducibility metadata.
- `<lib>/stage4_results/results.csv`: per-API execution summary.
- `<lib>/stage4_results/execution_summary.csv`: base/valid/negative case-level audit.
- `<lib>/stage4_results/coverage_report.json`: coverage and limitation report.
- `<lib>/stage4_results/coverage_report.md`: human-readable coverage report.
- `<lib>/stage4_results/bug_report.csv`: candidate bug records.
- `<lib>/stage4_results/bug_report.json`: full candidate-bug report.
- `<lib>/stage4_results/failure.csv`: audit events.
- `<lib>/stage4_results/execution_results/*.results.json`: detailed per-API bundles.

What to show live:

1. Open `demo-run/api_list.txt`.
2. Open `demo-run/results/latest/jax/stage1_specs/jax.numpy.sin.json`.
3. Open `demo-run/results/latest/jax/stage3_init/jax.numpy.sin.init.json`.
4. Open `demo-run/results/latest/jax/generated_tests/test_jax_numpy_sin.py`.
5. Run the generated pytest command.
6. Open `demo-run/results/latest/demo_report.md`.
7. Open `demo-run/results/latest/torch/stage4_results/execution_summary.csv`.

## 14. What Are Frozen Tests?

"Frozen" means the API list was fixed before the run. It prevents accidental denominator drift.

Why this matters:

- If the API list changes during repair, results become unfair.
- A fixed list makes percentages meaningful.
- Generated pytest files correspond to that frozen list.

Can generated tests run on any existing code?

They can run against the installed library version/environment if:

- dependencies are installed
- the seed JSON paths exist
- the API still exists
- the behaviour is compatible with the saved seed

In this repo, generated tests call DeepFuzz's `fuzz_one` with an `.init.json` file. They are best thought of as replay/regression tests for the library environment, not generic unit tests for arbitrary application code. If you want to use them in another repo, regenerate them or adjust seed paths.

## 15. Why The Results Are Comparable To Related Work, But Not Identical

Fuzz4All:

- Fuzz4All uses LLMs as a universal fuzzing engine across systems and languages.
- It reported 98 bugs across systems such as compilers, solvers, Java, and Qiskit, with 64 confirmed as previously unknown.
- DeepFuzz is narrower, but more domain-grounded: it focuses on deep-learning Python APIs, documentation-derived specs, runtime validation, seeds, coverage, and triage artefacts.

VISTAFUZZ:

- VISTAFUZZ uses LLMs for document-guided fuzzing of OpenCV APIs.
- It tests 330 OpenCV APIs and reports 17 new bugs, with 10 confirmed and 5 fixed.
- DeepFuzz explores a similar document-guided idea, but in the harder DL-library setting with tensors, devices, dtypes, and Python/native backend boundaries.

DocTer:

- DocTer extracts constraints from DL API documentation and uses valid/invalid input generation.
- It reports 94 bugs from 174 API functions and 43 documentation inconsistencies.
- DeepFuzz follows the same motivation that documentation constraints are valuable, but uses a multi-stage artefact pipeline with local LLM repair, selected-function coverage, generated pytest export, and strict triage categories.

How to compare honestly:

- Do not claim DeepFuzz beats those systems.
- Say DeepFuzz is positioned differently: reproducible documentation-to-executable artefact pipeline for JAX, TensorFlow, and PyTorch.
- Your result is strong because it shows scale across 1,992 accepted APIs and produces auditable evidence, not because every candidate bug is already confirmed.

Sources:

- Fuzz4All: https://arxiv.org/abs/2308.04748
- VISTAFUZZ: https://arxiv.org/abs/2507.14558
- DocTer: https://arxiv.org/abs/2109.01002

## 16. What Was Good And What Was Bad

Good:

- Complete end-to-end implementation.
- Works across three major DL libraries.
- Produces reproducible artefacts.
- Uses deterministic seeds and generated pytest replay.
- Separates bug candidates from documentation mismatches and pipeline failures.
- Uses selected-function coverage instead of misleading whole-library coverage.
- Supports device oracle and edge numeric mutations.

Bad or limited:

- Native/kernel coverage is unavailable from binary wheels.
- Some APIs are filtered out because they are not safe one-call targets.
- Some generated specs/seeds fail validation or smoke tests.
- TensorFlow coverage is harder because of wrappers and source attribution.
- PyTorch mismatch candidates need manual reduction.
- LLM extraction requires validation and repair; it is not reliable by itself.

How to defend this:

- "The limitations are exactly why the pipeline has strict triage."
- "A weaker project would hide invalid cases; DeepFuzz records them."
- "Completeness here means a working, auditable pipeline, not perfect coverage of every native kernel."

## 17. Presentation Script By Poster Section

### Title

Say:

"My project is DeepFuzz, a documentation-driven fuzzing pipeline for deep-learning Python APIs."

Short bullet:

- "Docs in, executable fuzzing evidence out."

### Motivation

Say:

"Deep-learning libraries are large, fast moving, and difficult to test manually. Documentation describes intended behaviour, but documentation is not executable."

Short bullets:

- huge API surfaces
- docs are prose
- runtime behaviour depends on shapes, dtypes, devices

### Aim

Say:

"The aim was to turn public API documentation into deterministic fuzz tests and evidence across JAX, TensorFlow, and PyTorch."

Short bullets:

- specs
- seeds
- generated tests
- coverage
- triage

### Pipeline

Say:

"The pipeline has four main stages after API selection: documentation to specs, validation and repair, executable seed generation, and isolated fuzzing with coverage and triage."

Short bullets:

- docs -> JSON
- JSON -> validated specs
- specs -> seeds
- seeds -> fuzz runs
- runs -> reports/tests

### Results

Say:

"The final runs accepted 1,992 APIs and generated 1,407 pytest tests. API execution coverage was high: 100% for JAX, 98.8652% for TensorFlow, and 100% for PyTorch on evaluated APIs."

Short bullets:

- scale
- high base-valid execution
- selected-function coverage differs by library
- candidate bugs are conservative

### Triage

Say:

"A key engineering contribution is strict triage. Negative rejections are not bugs. Candidate bugs, documentation mismatches, pipeline failures, and device mismatches are separate."

Short bullets:

- no inflated bug claims
- reproducible evidence
- manual confirmation required

### Demo

Say:

"For the live demo I run a six-API subset, two APIs from each library, and show the artefact trail from spec to seed to pytest to Stage 4 report."

Short bullets:

- type APIs
- run mini pipeline
- open seed
- run pytest
- open report

### Conclusion

Say:

"DeepFuzz demonstrates a complete and reusable workflow for turning API documentation into executable fuzzing evidence."

Short bullets:

- complete
- functional
- reproducible
- useful beyond bug finding

## 18. Examiner Q&A

**Why not use whole-library coverage?**
Because whole-library coverage for large ML libraries mostly measures package size and import structure, not whether selected APIs were exercised. Selected-function coverage is more honest.

**Why are some coverage numbers low?**
Some APIs are wrappers, delegate into native kernels, or have source spans that are hard to attribute. Low selected-function coverage can mean the API wrapper is thin, not that the test is useless.

**Why not test every API?**
Because many public APIs are unsafe or unsuitable as one-call fuzz targets. Filtering makes the denominator meaningful and reduces false positives.

**Are all candidate bugs real?**
No. They are candidates. They need reproduction, reduction, version checks, and manual review.

**What is the main innovation?**
The combination of documentation-first extraction, runtime-grounded validation, executable seeds, generated tests, selected-function coverage, and strict triage.

**What is useful besides bug finding?**
Regression testing, documentation validation, API migration checks, CPU/GPU consistency checks, and reproducible test generation.

**How would you extend this to another library?**
Write an adapter for API collection, import rules, tensor/dtype/device materialisation, device movement, coverage source config, and environment exclusions. The core pipeline remains the same.

## 19. One-Minute Plain-English Explanation

"DeepFuzz starts with API documentation. It extracts a structured JSON spec for each API, validates that spec, then turns it into a deterministic seed input. If the seed runs, DeepFuzz mutates it, runs the API in isolated workers, records coverage, and writes bug/triage reports and generated pytest tests. The key idea is that documentation gives intended behaviour, but runtime validation decides what is actually executable. My thesis shows this pipeline working across JAX, TensorFlow, and PyTorch, with conservative bug reporting and reproducible artefacts at every stage."

## 20. What To Emphasise For Marks

Poster 15%:

- clean structure
- readable flow
- diagrams and result table
- clear scope and background

Presentation 20%:

- explain one API through the pipeline
- answer questions with artefacts
- be honest about limitations

Excellence/Innovation 35%:

- docs-to-executable pipeline
- validation/repair
- selected-function coverage
- device oracle
- strict triage

Completeness 30%:

- implemented stages
- working commands
- final reports
- generated tests
- demo-run artefacts
