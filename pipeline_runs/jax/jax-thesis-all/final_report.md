# DeepFuzz Final Report

DeepFuzz reports selected-function Python line coverage over the APIs that reached Stage 4. Accepted documented APIs, frozen run-list entries, and evaluated/base-valid APIs are reported separately so the funnel remains auditable.

- Library: jax
- Frozen APIs: 379
- Accepted documented APIs: 477
- Stage 2 valid specs: 477/477 (100.0%)
- Stage 3 ready seeds: 476/477 (99.7904%)
- Stage 4 evaluated APIs: 476/477 (99.7904%) accepted; 476/379 (125.5937%) frozen
- Coverage metric: 52.3032 (Selected function line coverage %)
- Coverage method: python_coverage
- Base-valid APIs: 476/477 (99.7904%) accepted; 476/476 (100.0%) evaluated
- API execution coverage over evaluated APIs: 476/476 (100.0%)
- Selected-function line coverage: 52.3032
- Package Python coverage: 0.0121
- Total valid programs: 3421
- Unique valid programs: 2718
- Expected negative rejections: 88
- Invalid valid mutations: 13
- Device oracle mismatches: 0
- Negative accepted not bug: 874
- Candidate implementation bugs: 2
- Candidate documentation mismatches: 0
- Pipeline errors: 2
- Platform/runtime exclusions: 1
- Execution time: 19343.4077 seconds

## API Funnel

| Stage | APIs | Notes |
| --- | ---: | --- |
| Accepted documented APIs | 477 | Baseline denominator from `/Users/aryansg/Desktop/DeepFuzz/doc2info/results/jax/accepted.csv` |
| Frozen API list entries | 379 | Entries in `/Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/api_list.txt` |
| Stage 2 valid specs | 477 | 0 Stage 2 failures |
| Stage 3 ready seeds | 476 | 0 seed/init failures; 1 platform/runtime exclusions |
| Frozen entries ready for Stage 4 | 379 | 97 Stage 3-ready APIs are outside the frozen list |
| Stage 4 evaluated APIs | 476 | 0 frozen internal APIs excluded before Stage 4 |
| Stage 4 base-valid APIs | 476 | 0 non-internal frozen ready APIs not executed |

## Final Evaluation Table

| Field | Value |
| --- | --- |
| Library | jax |
| Frozen APIs | 379 |
| Accepted documented APIs | 477 |
| Stage 2 valid specs | 477/477 (100.0%) |
| Stage 2 failures | 0 |
| Input API list entries | 379 |
| Ready seeds | 476/477 (99.7904%) |
| Frozen entries ready for Stage 4 | 379/379 (100.0%) |
| Stage 3 non-ready APIs | 1 |
| Platform/runtime exclusions | 1 |
| Stage 3 ready APIs outside frozen list | 97 |
| Stage 4 evaluated APIs | 476/477 (99.7904%) accepted; 476/379 (125.5937%) frozen |
| Internal frozen APIs excluded before Stage 4 | 0 |
| Non-internal frozen ready APIs not executed | 0 |
| Base-valid APIs | 476/477 (99.7904%) accepted; 476/476 (100.0%) evaluated |
| Accepted API execution coverage | 99.7904 |
| Frozen API execution coverage | 125.5937 |
| Evaluated API execution coverage | 100.0 |
| Generated test cases | 379 |
| Valid programs | 3421 |
| Expected negative rejections | 88 |
| Invalid valid mutations | 13 |
| Device oracle mismatches | 0 |
| Negative accepted not bug | 874 |
| Candidate doc mismatches | 0 |
| Candidate implementation bugs | 2 |
| API execution coverage | 100.0 |
| Selected-function line coverage | 52.3032 |
| Package Python coverage | 0.0121 |
| Runtime | 19343.4077 |

## Artifacts
- Coverage JSON: stage4/results/jax-thesis-all-coverage/coverage_report.json
- Selected-function coverage JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/coverage/selected_function_coverage.json
- Bug JSON: stage4/results/jax-thesis-all-coverage/bug_report.json
- Bug CSV: stage4/results/jax-thesis-all-coverage/bug_report.csv
- Triage: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage
- Stage 3 seed failures: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage/stage3_seed_failures.csv
- Stage 3 platform exclusions: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage/stage3_platform_exclusions.csv
- Accepted APIs missing Stage 3: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage/accepted_not_stage3.txt
- Internal APIs excluded before Stage 4: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage/internal_excluded_before_stage4.txt
- Stage 3-ready APIs outside frozen list: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/triage/stage3_ready_outside_frozen.txt
- Final JSON: /Users/aryansg/Desktop/DeepFuzz/pipeline_runs/jax/jax-thesis-all/final_report.json

## Coverage Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
