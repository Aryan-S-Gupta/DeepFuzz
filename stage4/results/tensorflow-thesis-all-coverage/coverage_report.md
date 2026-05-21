# DeepFuzz Stage 4 Coverage Summary

- Target: tensorflow 2.21.0
- Coverage metric: 31.2415 (Selected function line coverage %)
- Method: python_coverage
- API execution coverage: 697/705 (98.8652%)
- Unique valid programs: 6866
- Candidate bugs: 3
- APIs below low-coverage threshold: 238
- Execution time: 59920.88 seconds
- Line coverage: 31.2415% (1827/5848 lines)
- Selected-function line coverage: 31.2415%

## Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
