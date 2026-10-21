#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/LSTM_5DS_Custom"
PYTHON="/root/miniconda3/bin/python"
cd "$PROJECT_ROOT"
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  code=$?
  printf 'stage: failed\nexit_code: %s\ntimestamp: %s\n' "$code" "$(date -u +%FT%TZ)" > LIVE_STATUS.txt
  printf 'FAILED exit=%s utc=%s\n' "$code" "$(date -u +%FT%TZ)" > logs/pipeline_status.txt
  exit "$code"
}
trap on_error ERR

printf 'RUNNING data_generation utc=%s\n' "$(date -u +%FT%TZ)" > logs/pipeline_status.txt
"$PYTHON" -u code/generate_data.py 2>&1 | tee logs/data_generation.log
printf 'RUNNING experiments utc=%s\n' "$(date -u +%FT%TZ)" > logs/pipeline_status.txt
"$PYTHON" -u code/pipeline.py 2>&1 | tee logs/pipeline.log
printf 'COMPLETE utc=%s\n' "$(date -u +%FT%TZ)" > logs/pipeline_status.txt
