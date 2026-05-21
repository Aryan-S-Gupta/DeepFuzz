#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  LIB="$1"
  shift
else
  LIB="${LIB:-jax}"
fi

RUN_ID="${RUN_ID:-${LIB}-thesis-all}"
SEED="${SEED:-1337}"
MUTATION_BUDGET="${MUTATION_BUDGET:-32}"
COVERAGE_SCOPE="${COVERAGE_SCOPE:-python}"
ENABLE_PYTHON_COVERAGE="${ENABLE_PYTHON_COVERAGE:-1}"
PYTHON_COV_SOURCE="${PYTHON_COV_SOURCE:-$LIB}"
LOW_COVERAGE_THRESHOLD="${LOW_COVERAGE_THRESHOLD:-10}"
PYTHON="${PYTHON:-python3}"
INIT_DIR="${INIT_DIR:-json2init/results/${LIB}}"
OK_CSV="${OK_CSV:-${INIT_DIR}/ok.csv}"
STAGE4_RESULTS_DIR="${STAGE4_RESULTS_DIR:-stage4/results/${LIB}-thesis-all-coverage}"
RUN_DIR="pipeline_runs/${LIB}/${RUN_ID}"
FROZEN_API_LIST="${THESIS_API_LIST:-${RUN_DIR}/api_list.txt}"
LOW_API_LIST="${LOW_API_LIST:-${RUN_DIR}/repair/low_coverage_api_list.txt}"

if [[ ! -f "$FROZEN_API_LIST" ]]; then
  echo "[lowcov] missing frozen API list: $FROZEN_API_LIST" >&2
  exit 1
fi

"$PYTHON" scripts/snapshot_run.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR" --label "before-lowcov-b${MUTATION_BUDGET}"

"$PYTHON" scripts/select_low_coverage_apis.py \
  --lib "$LIB" \
  --run-id "$RUN_ID" \
  --stage4-results-dir "$STAGE4_RESULTS_DIR" \
  --api-list "$FROZEN_API_LIST" \
  --threshold "$LOW_COVERAGE_THRESHOLD" \
  --out "$LOW_API_LIST" \
  "$@"

if [[ ! -s "$LOW_API_LIST" ]]; then
  echo "[lowcov] no APIs below threshold ${LOW_COVERAGE_THRESHOLD}; refreshing report only"
  "$PYTHON" scripts/report.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR"
  "$PYTHON" scripts/snapshot_run.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR" --label "no-lowcov-b${MUTATION_BUDGET}"
  exit 0
fi

echo "[lowcov] rerunning $(wc -l < "$LOW_API_LIST" | tr -d ' ') APIs from $LOW_API_LIST"

LIB="$LIB" \
RUN_ID="$RUN_ID" \
SEED="$SEED" \
MUTATION_BUDGET="$MUTATION_BUDGET" \
COVERAGE_SCOPE="$COVERAGE_SCOPE" \
ENABLE_PYTHON_COVERAGE="$ENABLE_PYTHON_COVERAGE" \
PYTHON_COV_SOURCE="$PYTHON_COV_SOURCE" \
INIT_DIR="$INIT_DIR" \
OK_CSV="$OK_CSV" \
STAGE4_RESULTS_DIR="$STAGE4_RESULTS_DIR" \
PYTHON="$PYTHON" \
STAGE4_MERGE_UNSELECTED=1 \
  ./scripts/run_stage4.sh "$LIB" --only-api-list "$LOW_API_LIST" --merge-unselected

"$PYTHON" scripts/report.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR"
"$PYTHON" scripts/snapshot_run.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR" --label "after-lowcov-b${MUTATION_BUDGET}"
"$PYTHON" scripts/read_coverage.py --stage4-results-dir "$STAGE4_RESULTS_DIR"
