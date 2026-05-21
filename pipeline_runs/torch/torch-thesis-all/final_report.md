# DeepFuzz Final Report

DeepFuzz reports selected-function Python line coverage over the APIs that reached Stage 4. Accepted documented APIs, frozen run-list entries, and evaluated/base-valid APIs are reported separately so the funnel remains auditable.

- Library: torch
- Frozen APIs: 323
- Accepted documented APIs: 593
- Stage 2 valid specs: 593/593 (100.0%)
- Stage 3 ready seeds: 323/593 (54.4688%)
- Stage 4 evaluated APIs: 323/593 (54.4688%) accepted; 323/323 (100.0%) frozen
- Coverage metric: 59.5745 (Selected function line coverage %)
- Coverage method: python_coverage
- Base-valid APIs: 323/593 (54.4688%) accepted; 323/323 (100.0%) evaluated
- API execution coverage over evaluated APIs: 323/323 (100.0%)
- Selected-function line coverage: 59.5745
- Package Python coverage: 0.0032
- Total valid programs: 2096
- Unique valid programs: 2096
- Expected negative rejections: 51
- Invalid valid mutations: 7
- Device oracle mismatches: 1209
- Negative accepted not bug: 447
- Candidate implementation bugs: 185
- Candidate documentation mismatches: 0
- Pipeline errors: 265
- Platform/runtime exclusions: 19
- Execution time: 39238.8155 seconds

## API Funnel

| Stage | APIs | Notes |
| --- | ---: | --- |
| Accepted documented APIs | 593 | Baseline denominator from `/Users/aryansg/Desktop/DeepFuzz/doc2info/results/torch/accepted.csv` |
| Frozen API list entries | 323 | Entries in `/Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/api_list.txt` |
| Stage 2 valid specs | 593 | 0 Stage 2 failures |
| Stage 3 ready seeds | 323 | 251 seed/init failures; 19 platform/runtime exclusions |
| Frozen entries ready for Stage 4 | 323 | 0 Stage 3-ready APIs are outside the frozen list |
| Stage 4 evaluated APIs | 323 | 0 frozen internal APIs excluded before Stage 4 |
| Stage 4 base-valid APIs | 323 | 0 non-internal frozen ready APIs not executed |

## Final Evaluation Table

| Field | Value |
| --- | --- |
| Library | torch |
| Frozen APIs | 323 |
| Accepted documented APIs | 593 |
| Stage 2 valid specs | 593/593 (100.0%) |
| Stage 2 failures | 0 |
| Input API list entries | 323 |
| Ready seeds | 323/593 (54.4688%) |
| Frozen entries ready for Stage 4 | 323/323 (100.0%) |
| Stage 3 non-ready APIs | 270 |
| Platform/runtime exclusions | 19 |
| Stage 3 ready APIs outside frozen list | 0 |
| Stage 4 evaluated APIs | 323/593 (54.4688%) accepted; 323/323 (100.0%) frozen |
| Internal frozen APIs excluded before Stage 4 | 0 |
| Non-internal frozen ready APIs not executed | 0 |
| Base-valid APIs | 323/593 (54.4688%) accepted; 323/323 (100.0%) evaluated |
| Accepted API execution coverage | 54.4688 |
| Frozen API execution coverage | 100.0 |
| Evaluated API execution coverage | 100.0 |
| Generated test cases | 323 |
| Valid programs | 2096 |
| Expected negative rejections | 51 |
| Invalid valid mutations | 7 |
| Device oracle mismatches | 1209 |
| Negative accepted not bug | 447 |
| Candidate doc mismatches | 0 |
| Candidate implementation bugs | 185 |
| API execution coverage | 100.0 |
| Selected-function line coverage | 59.5745 |
| Package Python coverage | 0.0032 |
| Runtime | 39238.8155 |

## Artifacts
- Coverage JSON: stage4/results/torch-thesis-all-coverage/coverage_report.json
- Selected-function coverage JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/coverage/selected_function_coverage.json
- Bug JSON: stage4/results/torch-thesis-all-coverage/bug_report.json
- Bug CSV: stage4/results/torch-thesis-all-coverage/bug_report.csv
- Triage: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage
- Stage 3 seed failures: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage/stage3_seed_failures.csv
- Stage 3 platform exclusions: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage/stage3_platform_exclusions.csv
- Accepted APIs missing Stage 3: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage/accepted_not_stage3.txt
- Internal APIs excluded before Stage 4: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage/internal_excluded_before_stage4.txt
- Stage 3-ready APIs outside frozen list: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/triage/stage3_ready_outside_frozen.txt
- Final JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/torch/torch-thesis-all/final_report.json

## Coverage Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
