#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/Chaos_2Systems_3Models/kolmogorov2d"
PYTHON="/root/miniconda3/bin/python"
STATUS="$PROJECT_ROOT/server_logs/easy_attention_sgu_v2_promoted95_status.txt"
PREDICTION_LOG="$PROJECT_ROOT/server_logs/easy_attention_sgu_v2_promoted95_prediction.log"
FINALIZATION_LOG="$PROJECT_ROOT/server_logs/easy_attention_sgu_v2_promoted95_finalization.log"

cd "$PROJECT_ROOT"

on_error() {
  code=$?
  printf 'FAILED stage=%s exit=%s utc=%s\n' "${STAGE:-unknown}" "$code" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$code"
}
trap on_error ERR

STAGE=prediction
printf 'RUNNING stage=prediction checkpoint=promoted_epoch95 utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/predict_best_nrmse.py \
  --model easy_attention_sgu --ds 1 --sim-time 5000 --pred-len 100 \
  --steps 2000 --trajectories 15 --checkpoint best_rollout \
  2>&1 | tee "$PREDICTION_LOG"

STAGE=finalization
printf 'RUNNING stage=finalization checkpoint=promoted_epoch95 utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/finalize_results.py --model easy_attention_sgu \
  2>&1 | tee "$FINALIZATION_LOG"

printf 'COMPLETE checkpoint=promoted_epoch95 utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
