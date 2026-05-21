# DeepFuzz Stage 4 Coverage Summary

- Target: torch 2.11.0
- Coverage metric: 59.5745 (Selected function line coverage %)
- Method: python_coverage
- API execution coverage: 323/323 (100.0%)
- Unique valid programs: 2096
- Candidate bugs: 185
- APIs below low-coverage threshold: 39
- Execution time: 39238.8155 seconds
- Line coverage: 59.5745% (336/564 lines)
- Selected-function line coverage: 59.5745%

## Limitations
- Python coverage measures Python wrapper/harness lines and may not represent native kernel coverage.
- Selected-function coverage is aggregated from merged per-API Python coverage JSON rows when available, so targeted repair reruns keep coverage for APIs outside the rerun subset.
- Native target line coverage requires an instrumented source build with coverage data; no fake native lines are emitted.
