# DeepFuzz Stage 4 Coverage Summary

- Target: jax 0.10.0
- Coverage metric: 52.3032 (Selected function line coverage %)
- Method: python_coverage
- API execution coverage: 476/476 (100.0%)
- Unique valid programs: 2718
- Candidate bugs: 2
- APIs below low-coverage threshold: 211
- Execution time: 19343.4077 seconds
- Line coverage: 52.3032% (1249/2388 lines)
- Selected-function line coverage: 52.3032%

## Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
