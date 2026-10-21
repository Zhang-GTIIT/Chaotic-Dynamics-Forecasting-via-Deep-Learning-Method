#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/root/miniconda3/bin/python"
STATUS="$PROJECT_ROOT/server_logs/pipeline_status.json"
LOCK="$PROJECT_ROOT/server_logs/pipeline.lock"
MASTER_LOG="$PROJECT_ROOT/server_logs/pipeline.log"
mkdir -p "$PROJECT_ROOT/server_logs"
cd "$PROJECT_ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another pipeline instance holds $LOCK" >&2
  exit 73
fi

SYSTEM="$($PYTHON -c "import yaml; print(yaml.safe_load(open('code/config.yaml'))['experiment']['system'])")"
SIM_TIME="$($PYTHON -c "import yaml; print(yaml.safe_load(open('code/config.yaml'))['data_generation']['sim_time'])")"
STEPS="$($PYTHON -c "import yaml; print(yaml.safe_load(open('code/config.yaml'))['prediction']['total_pred_steps'])")"
NUM_EVAL="$($PYTHON -c "import yaml; print(yaml.safe_load(open('code/config.yaml'))['prediction']['num_eval_trajectories'])")"
STAGE="initializing"

write_status() {
  local state="$1"
  local message="$2"
  local temporary="${STATUS}.tmp"
  "$PYTHON" - "$temporary" "$STATUS" "$SYSTEM" "$state" "$STAGE" "$message" <<'PY'
import json, os, sys
from datetime import datetime, timezone
temporary, final, system, state, stage, message = sys.argv[1:]
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "system": system,
            "state": state,
            "stage": stage,
            "message": message,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        handle,
        indent=2,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, final)
PY
}

on_error() {
  local code=$?
  write_status "failed" "command failed with exit code $code"
  exit "$code"
}
trap on_error ERR
write_status "running" "pipeline started"

STAGE="data_generation_and_validation"
write_status "running" "generating or validating independent datasets"
"$PYTHON" -u code/generate_data.py 2>&1 | tee -a server_logs/data_generation.log
"$PYTHON" -u code/main.py --check-data --ds 1 --sim-time "$SIM_TIME" 2>&1 | tee -a server_logs/data_generation.log

STAGE="model_smoke_check"
write_status "running" "checking system-adapted model and rollout shapes"
"$PYTHON" -u code/smoke_check.py 2>&1 | tee -a server_logs/smoke_check.log

STAGE="gpu_memory_probe"
write_status "running" "checking full-horizon configured batch sizes"
"$PYTHON" -u code/resource_probe.py 2>&1 | tee -a server_logs/resource_probe.log

for MODEL in easy_attention_sgu self_attention lstm; do
  STAGE="${MODEL}_training"
  write_status "running" "training $MODEL"
  "$PYTHON" -u code/main.py     --model "$MODEL" --ds 1 --sim-time "$SIM_TIME" --pred-len 100 --resume     2>&1 | tee -a "server_logs/${MODEL}_training.log"

  STAGE="${MODEL}_prediction"
  write_status "running" "running independent-test prediction for $MODEL"
  "$PYTHON" -u code/predict_best_nrmse.py     --model "$MODEL" --ds 1 --sim-time "$SIM_TIME" --pred-len 100     --steps "$STEPS" --trajectories "$NUM_EVAL" --checkpoint best_rollout     2>&1 | tee -a "server_logs/${MODEL}_prediction.log"

  STAGE="${MODEL}_finalization"
  write_status "running" "creating compact metrics and figures for $MODEL"
  "$PYTHON" -u code/finalize_results.py --model "$MODEL"     2>&1 | tee -a "server_logs/${MODEL}_finalization.log"
done

STAGE="three_model_comparison"
write_status "running" "creating three-model comparison"
"$PYTHON" -u code/compare_system.py 2>&1 | tee -a server_logs/comparison.log

STAGE="delivery"
write_status "running" "assembling checkpoint-free desktop delivery"
"$PYTHON" -u code/make_delivery.py 2>&1 | tee -a server_logs/delivery.log

STAGE="complete"
write_status "complete" "all three models, predictions, figures, comparison, and delivery completed"
touch "$PROJECT_ROOT/server_logs/PIPELINE_COMPLETE"
echo "$SYSTEM pipeline complete" | tee -a "$MASTER_LOG"
