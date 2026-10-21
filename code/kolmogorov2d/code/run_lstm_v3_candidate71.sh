#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/Chaos_2Systems_3Models/kolmogorov2d"
PYTHON="/root/miniconda3/bin/python"
MODEL_DIR="$PROJECT_ROOT/models/lstm/ds1/t5000/s128_p100"
STATUS="$PROJECT_ROOT/server_logs/lstm_v3_candidate71_status.txt"

cd "$PROJECT_ROOT"
cp "$MODEL_DIR/latest.pt" "$MODEL_DIR/latest_epoch74_lstm_v3.pt"
cp "$MODEL_DIR/candidate_epoch70_lstm_v3.pt" "$MODEL_DIR/latest.pt"
restore_latest() {
  cp "$MODEL_DIR/latest_epoch74_lstm_v3.pt" "$MODEL_DIR/latest.pt"
}
trap restore_latest EXIT

printf 'RUNNING utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
"$PYTHON" -u code/predict_best_nrmse.py \
  --model lstm --ds 1 --sim-time 5000 --pred-len 100 \
  --steps 2000 --trajectories 15 --checkpoint latest \
  > server_logs/lstm_v3_candidate71_prediction.log 2>&1
printf 'COMPLETE utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
