#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/Chaos_2Systems_3Models/kolmogorov2d"
PYTHON="/root/miniconda3/bin/python"
STATUS="$PROJECT_ROOT/server_logs/lstm_v2_status.txt"
TRAIN_LOG="$PROJECT_ROOT/server_logs/lstm_v2_training.log"
PREDICTION_LOG="$PROJECT_ROOT/server_logs/lstm_v2_prediction.log"
FINALIZATION_LOG="$PROJECT_ROOT/server_logs/lstm_v2_finalization.log"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/server_logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  code=$?
  printf 'FAILED stage=%s exit=%s utc=%s\n' "${STAGE:-unknown}" "$code" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$code"
}
trap on_error ERR

STAGE=training
printf 'RUNNING stage=training utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/main.py \
  --model lstm --ds 1 --sim-time 5000 --pred-len 100 --no-resume \
  2>&1 | tee "$TRAIN_LOG"

STAGE=prediction
printf 'RUNNING stage=prediction utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/predict_best_nrmse.py \
  --model lstm --ds 1 --sim-time 5000 --pred-len 100 \
  --steps 2000 --trajectories 15 --checkpoint best_rollout \
  2>&1 | tee "$PREDICTION_LOG"

STAGE=finalization
printf 'RUNNING stage=finalization utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/finalize_results.py --model lstm \
  2>&1 | tee "$FINALIZATION_LOG"

printf 'COMPLETE utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
