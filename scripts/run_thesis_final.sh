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
PYTHON="${PYTHON:-python3}"
THESIS_LIMIT="${THESIS_LIMIT:-0}"
SEED="${SEED:-1337}"
MUTATION_BUDGET="${MUTATION_BUDGET:-8}"
COVERAGE_SCOPE="${COVERAGE_SCOPE:-python}"
ENABLE_PYTHON_COVERAGE="${ENABLE_PYTHON_COVERAGE:-1}"
PYTHON_COV_SOURCE="${PYTHON_COV_SOURCE:-$LIB}"
INIT_DIR="${INIT_DIR:-json2init/results/${LIB}}"
SPEC_DIR="${SPEC_DIR:-info2json/results/${LIB}}"
VALIDATOR_OK_CSV="${VALIDATOR_OK_CSV:-json_validator/results/${LIB}/ok.csv}"
STAGE4_RESULTS_DIR="${STAGE4_RESULTS_DIR:-stage4/results/${LIB}-thesis-all-coverage}"
RUN_DIR="pipeline_runs/${LIB}/${RUN_ID}"
RUN_API_LIST="${RUN_DIR}/api_list.txt"

mkdir -p "$RUN_DIR" "${RUN_DIR}/generated_tests" "${RUN_DIR}/coverage/html" "${RUN_DIR}/triage" "${RUN_DIR}/repro"

if [[ -n "${THESIS_API_LIST:-}" ]]; then
  if [[ ! -f "$THESIS_API_LIST" ]]; then
    echo "[thesis] THESIS_API_LIST does not exist: $THESIS_API_LIST" >&2
    exit 1
  fi
  if [[ ! -f "$RUN_API_LIST" ]] || ! cmp -s "$THESIS_API_LIST" "$RUN_API_LIST"; then
    cp "$THESIS_API_LIST" "$RUN_API_LIST"
  fi
elif [[ ! -f "$RUN_API_LIST" ]]; then
  echo "[thesis] missing frozen api_list.txt. Run scripts/select_api_subset.py first or set THESIS_API_LIST." >&2
  exit 1
fi

api_hash_before="$("$PYTHON" - "$RUN_API_LIST" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"

echo "[thesis] library=${LIB} run_id=${RUN_ID} seed=${SEED} budget=${MUTATION_BUDGET} coverage=${COVERAGE_SCOPE}"
echo "[thesis] frozen API list: ${RUN_API_LIST}"

if [[ "$ENABLE_PYTHON_COVERAGE" == "1" ]]; then
  "$PYTHON" - <<'PY'
try:
    import coverage  # noqa: F401
except Exception as exc:
    raise SystemExit(f"coverage.py is required for source-line coverage in this command: {type(exc).__name__}: {exc}")
PY
fi

stage3_args=(
  "$PYTHON" json2init/json2init.py
  --spec-dir "$SPEC_DIR"
  --outdir "$INIT_DIR"
  --ok-csv "$VALIDATOR_OK_CSV"
  --smoke-test
  --reuse-existing
  --only-api-list "$RUN_API_LIST"
)
if [[ "$THESIS_LIMIT" != "0" ]]; then
  stage3_args+=(--limit "$THESIS_LIMIT")
fi
"${stage3_args[@]}"

"$PYTHON" scripts/export_generated_tests.py \
  --lib "$LIB" \
  --run-id "$RUN_ID" \
  --api-list "$RUN_API_LIST" \
  --out "${RUN_DIR}/generated_tests" \
  --seed "$SEED"

"$PYTHON" scripts/write_run_manifest.py \
  --lib "$LIB" \
  --run-id "$RUN_ID" \
  --api-list "$RUN_API_LIST" \
  --out "${RUN_DIR}/run_manifest.json" \
  --seed "$SEED" \
  --mutation-budget "$MUTATION_BUDGET" \
  --coverage-scope "$COVERAGE_SCOPE" \
  --native-coverage "${NATIVE_COVERAGE_STATUS:-unavailable_on_this_run}"

stage4_extra=(--only-api-list "$RUN_API_LIST")
if [[ "$THESIS_LIMIT" != "0" ]]; then
  stage4_extra+=(--limit "$THESIS_LIMIT")
fi

LIB="$LIB" \
RUN_ID="$RUN_ID" \
SEED="$SEED" \
MUTATION_BUDGET="$MUTATION_BUDGET" \
COVERAGE_SCOPE="$COVERAGE_SCOPE" \
ENABLE_PYTHON_COVERAGE="$ENABLE_PYTHON_COVERAGE" \
PYTHON_COV_SOURCE="$PYTHON_COV_SOURCE" \
STAGE4_RESULTS_DIR="$STAGE4_RESULTS_DIR" \
PYTHON="$PYTHON" \
  ./scripts/run_stage4.sh "$LIB" "${stage4_extra[@]}" "$@"

"$PYTHON" scripts/report.py --lib "$LIB" --run-id "$RUN_ID" --stage4-results-dir "$STAGE4_RESULTS_DIR"

api_hash_after="$("$PYTHON" - "$RUN_API_LIST" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"
if [[ "$api_hash_before" != "$api_hash_after" ]]; then
  echo "[thesis] frozen API list changed during run; refusing to continue" >&2
  exit 1
fi

echo "[thesis] coverage JSON: ${STAGE4_RESULTS_DIR}/coverage_report.json"
echo "[thesis] selected-function coverage: ${RUN_DIR}/coverage/selected_function_coverage.json"
echo "[thesis] execution summary: ${STAGE4_RESULTS_DIR}/execution_summary.csv"
echo "[thesis] failures/audit: ${STAGE4_RESULTS_DIR}/failure.csv"
echo "[thesis] candidate bugs: ${STAGE4_RESULTS_DIR}/bug_report.csv"
echo "[thesis] final report: ${RUN_DIR}/final_report.md"
