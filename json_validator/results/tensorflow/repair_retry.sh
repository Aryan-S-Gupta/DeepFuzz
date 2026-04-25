#!/usr/bin/env bash
set -euo pipefail

python3 json_validator/json_validator.py --spec-dir info2json/results/tensorflow --api-csv doc2info/results/tensorflow/accepted.csv --state-dir json_validator/results/tensorflow --primary-repair-model mistral:7b --fallback-repair-model mixtral:8x7b --fallback-after-round 3 --repair-host http://localhost:11434 --max-rounds 3 --repair-timeout 300 --repair-num-predict 700 --repair-num-ctx 4096 --repair-temperature 0.0 --only-apis json_validator/results/tensorflow/retry_api_list.txt
