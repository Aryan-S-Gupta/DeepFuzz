#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${1:-/workspace/DeepFuzz}
cd /workspace

if [ ! -d jax ]; then
  git clone https://github.com/jax-ml/jax.git
fi
cd jax
python3 -m pip install jaxlib
python3 -m pip install -e .

cd "$REPO_ROOT"
python3 stage4/stage4_coverage_runner.py \
  --init-dir json2init/results/jax \
  --results-dir stage4/results/jax \
  --ok-csv json2init/results/jax/ok.csv \
  --mutation-budget 8 \
  --seed 1337 \
  --coverage-scope python \
  --enable-python-coverage \
  --python-cov-source jax \
  --low-coverage-threshold 60
