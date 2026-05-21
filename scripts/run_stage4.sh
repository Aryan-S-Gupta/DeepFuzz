#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  LIB="$1"
  shift
else
  LIB="${LIB:-torch}"
fi
INIT_DIR="${INIT_DIR:-json2init/results/${LIB}}"
OK_CSV="${OK_CSV:-${INIT_DIR}/ok.csv}"
RESULTS_DIR="${STAGE4_RESULTS_DIR:-stage4/results/${LIB}-coverage}"
COVERAGE_SCOPE="${COVERAGE_SCOPE:-both}"
MUTATION_BUDGET="${MUTATION_BUDGET:-8}"
SEED="${SEED:-1337}"
RUN_ID="${RUN_ID:-}"
WORKER_TIMEOUT_SEC="${WORKER_TIMEOUT_SEC:-120}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-30}"
PYTHON="${PYTHON:-python3}"

cmd=("$PYTHON" stage4/stage4_coverage_runner.py --init-dir "$INIT_DIR" --results-dir "$RESULTS_DIR" --ok-csv "$OK_CSV" --mutation-budget "$MUTATION_BUDGET" --seed "$SEED" --run-id "$RUN_ID" --coverage-scope "$COVERAGE_SCOPE" --worker-timeout-sec "$WORKER_TIMEOUT_SEC" --case-timeout-sec "$CASE_TIMEOUT_SEC")

if [[ "${ENABLE_PYTHON_COVERAGE:-1}" == "1" ]]; then
  cmd+=(--enable-python-coverage --python-cov-source "${PYTHON_COV_SOURCE:-$LIB}")
fi
if [[ -n "${NATIVE_COVERAGE_ENGINE:-}" ]]; then
  cmd+=(--native-coverage-engine "$NATIVE_COVERAGE_ENGINE")
fi
if [[ -n "${NATIVE_SOURCE_ROOT:-}" ]]; then
  cmd+=(--native-source-root "$NATIVE_SOURCE_ROOT")
fi
if [[ -n "${NATIVE_BUILD_DIR:-}" ]]; then
  cmd+=(--native-build-dir "$NATIVE_BUILD_DIR")
fi
if [[ -n "${GCOVR_FILTERS:-}" ]]; then
  IFS=',' read -ra filters <<< "$GCOVR_FILTERS"
  for filt in "${filters[@]}"; do
    [[ -n "$filt" ]] && cmd+=(--gcovr-filter "$filt")
  done
fi
[[ "${NATIVE_HTML:-0}" == "1" ]] && cmd+=(--native-html)
[[ "${STAGE4_LIMIT:-0}" != "0" ]] && cmd+=(--limit "$STAGE4_LIMIT")
[[ -n "${KNOWN_BUGS_FILE:-}" ]] && cmd+=(--known-bugs-file "$KNOWN_BUGS_FILE")
[[ "${STAGE4_MERGE_UNSELECTED:-0}" == "1" ]] && cmd+=(--merge-unselected)
cmd+=("$@")

exec "${cmd[@]}"
