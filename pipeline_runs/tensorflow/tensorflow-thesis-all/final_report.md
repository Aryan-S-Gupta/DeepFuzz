# DeepFuzz Final Report

DeepFuzz reports selected-function Python line coverage over the APIs that reached Stage 4. Accepted documented APIs, frozen run-list entries, and evaluated/base-valid APIs are reported separately so the funnel remains auditable.

- Library: tensorflow
- Frozen APIs: 705
- Accepted documented APIs: 922
- Stage 2 valid specs: 904/922 (98.0477%)
- Stage 3 ready seeds: 705/922 (76.4642%)
- Stage 4 evaluated APIs: 705/922 (76.4642%) accepted; 705/705 (100.0%) frozen
- Coverage metric: 31.2415 (Selected function line coverage %)
- Coverage method: python_coverage
- Base-valid APIs: 697/922 (75.5965%) accepted; 697/705 (98.8652%) evaluated
- API execution coverage over evaluated APIs: 697/705 (98.8652%)
- Selected-function line coverage: 31.2415
- Package Python coverage: 0.1718
- Total valid programs: 6866
- Unique valid programs: 6866
- Expected negative rejections: 230
- Invalid valid mutations: 104
- Device oracle mismatches: 0
- Negative accepted not bug: 1298
- Candidate implementation bugs: 3
- Candidate documentation mismatches: 0
- Pipeline errors: 218
- Platform/runtime exclusions: 0
- Execution time: 59920.88 seconds

## API Funnel

| Stage | APIs | Notes |
| --- | ---: | --- |
| Accepted documented APIs | 922 | Baseline denominator from `/Users/aryansg/Desktop/DeepFuzz/doc2info/results/tensorflow/accepted.csv` |
| Frozen API list entries | 705 | Entries in `/Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/api_list.txt` |
| Stage 2 valid specs | 904 | 6 Stage 2 failures |
| Stage 3 ready seeds | 705 | 199 seed/init failures; 0 platform/runtime exclusions |
| Frozen entries ready for Stage 4 | 705 | 0 Stage 3-ready APIs are outside the frozen list |
| Stage 4 evaluated APIs | 705 | 0 frozen internal APIs excluded before Stage 4 |
| Stage 4 base-valid APIs | 697 | 0 non-internal frozen ready APIs not executed |

## Final Evaluation Table

| Field | Value |
| --- | --- |
| Library | tensorflow |
| Frozen APIs | 705 |
| Accepted documented APIs | 922 |
| Stage 2 valid specs | 904/922 (98.0477%) |
| Stage 2 failures | 6 |
| Input API list entries | 705 |
| Ready seeds | 705/922 (76.4642%) |
| Frozen entries ready for Stage 4 | 705/705 (100.0%) |
| Stage 3 non-ready APIs | 217 |
| Platform/runtime exclusions | 0 |
| Stage 3 ready APIs outside frozen list | 0 |
| Stage 4 evaluated APIs | 705/922 (76.4642%) accepted; 705/705 (100.0%) frozen |
| Internal frozen APIs excluded before Stage 4 | 0 |
| Non-internal frozen ready APIs not executed | 0 |
| Base-valid APIs | 697/922 (75.5965%) accepted; 697/705 (98.8652%) evaluated |
| Accepted API execution coverage | 75.5965 |
| Frozen API execution coverage | 98.8652 |
| Evaluated API execution coverage | 98.8652 |
| Generated test cases | 705 |
| Valid programs | 6866 |
| Expected negative rejections | 230 |
| Invalid valid mutations | 104 |
| Device oracle mismatches | 0 |
| Negative accepted not bug | 1298 |
| Candidate doc mismatches | 0 |
| Candidate implementation bugs | 3 |
| API execution coverage | 98.8652 |
| Selected-function line coverage | 31.2415 |
| Package Python coverage | 0.1718 |
| Runtime | 59920.88 |

## Artifacts
- Coverage JSON: stage4/results/tensorflow-thesis-all-coverage/coverage_report.json
- Selected-function coverage JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/coverage/selected_function_coverage.json
- Bug JSON: stage4/results/tensorflow-thesis-all-coverage/bug_report.json
- Bug CSV: stage4/results/tensorflow-thesis-all-coverage/bug_report.csv
- Triage: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage
- Stage 3 seed failures: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage/stage3_seed_failures.csv
- Stage 3 platform exclusions: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage/stage3_platform_exclusions.csv
- Accepted APIs missing Stage 3: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage/accepted_not_stage3.txt
- Internal APIs excluded before Stage 4: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage/internal_excluded_before_stage4.txt
- Stage 3-ready APIs outside frozen list: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/triage/stage3_ready_outside_frozen.txt
- Generated tests: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/generated_tests
- Final JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/tensorflow/tensorflow-thesis-all/final_report.json

## Coverage Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
