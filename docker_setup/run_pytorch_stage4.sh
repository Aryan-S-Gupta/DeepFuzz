#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${1:-/workspace/DeepFuzz}

python3 -c "import torch; print(torch.__version__); print(torch.__file__)"

cd "$REPO_ROOT"
python3 init2cov/stage4_coverage_runner.py \
  --init-dir json2init/results/torch \
  --results-dir init2cov/results/torch \
  --ok-csv json2init/results/torch/ok.csv \
  --mutation-budget 8 \
  --seed 1337 \
  --coverage-scope both \
  --enable-python-coverage \
  --python-cov-source torch \
  --native-coverage-engine gcovr \
  --native-source-root /workspace/pytorch \
  --native-build-dir /workspace/pytorch \
  --gcovr-filter '.*/aten/.*' \
  --gcovr-filter '.*/c10/.*' \
  --gcovr-filter '.*/torch/csrc/.*' \
  --low-coverage-threshold 60