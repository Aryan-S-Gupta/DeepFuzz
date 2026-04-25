#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${1:-/workspace/DeepFuzz}
cd /workspace

if [ ! -d tensorflow ]; then
  git clone https://github.com/tensorflow/tensorflow.git
fi
cd tensorflow
ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel || true
export PYTHON_BIN_PATH=$(which python3)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS='-O0 --coverage'
yes '' | python3 configure.py
bazel build --config=opt --copt=--coverage --copt=-O0 --linkopt=--coverage //tensorflow/tools/pip_package:wheel
python3 -m pip install $(find bazel-bin -name '*.whl' | head -n 1)

cd "$REPO_ROOT"
python3 stage4/stage4_coverage_runner.py \
  --init-dir json2init/results/tensorflow \
  --results-dir stage4/results/tensorflow \
  --ok-csv json2init/results/tensorflow/ok.csv \
  --mutation-budget 8 \
  --seed 1337 \
  --coverage-scope both \
  --enable-python-coverage \
  --python-cov-source tensorflow \
  --native-coverage-engine gcovr \
  --native-source-root /workspace/tensorflow \
  --native-build-dir /workspace/tensorflow/bazel-bin \
  --low-coverage-threshold 60
