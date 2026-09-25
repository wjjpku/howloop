#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wujiaju/LooPlus
PYTHON=/data/wujiaju/.venvs/loopreasoner/bin/python
RESULT_ROOT=results/boolean_dag_macrostep_seed0_retry2_20260710
LOG_ROOT=/data/wujiaju/logs

GPU_LIST="${GPU_LIST:-6}"
EXPECTED_PEAK_MIB="${EXPECTED_PEAK_MIB:-4096}"
RESERVE_MIB="${RESERVE_MIB:-16384}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

shared_gate_passes() {
  local gpu=$1 used total utilization free
  IFS=, read -r used total utilization < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
  )
  free=$((total - used))
  (( free >= EXPECTED_PEAK_MIB + RESERVE_MIB && utilization <= MAX_UTILIZATION ))
}

find_shared_gpu() {
  local gpu
  for gpu in $GPU_LIST; do
    if shared_gate_passes "$gpu"; then
      sleep 30
      if shared_gate_passes "$gpu"; then
        printf '%s\n' "$gpu"
        return 0
      fi
    fi
  done
  return 1
}

launch_condition() {
  local condition=$1
  local gpu=$2
  local session="macro_${condition}_train"
  local log="$LOG_ROOT/$session.log"
  local smoke="${condition}_batch512_smoke"
  local run="${condition}_seed0"
  local command
  command="exec > $log 2>&1; set -e; echo GPU_${gpu}_STARTING_${condition}_SMOKE; \
env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u -m reasoning_loop.boolean_dag_macrostep_train \
  --condition $condition --steps 1 --batch-size 512 --eval-batch-size 4 \
  --eval-batches 1 --eval-every 1 --print-every 1 --out-dir $RESULT_ROOT \
  --run-name $smoke; \
echo SMOKE_OK_STARTING_20K; \
env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u -m reasoning_loop.boolean_dag_macrostep_train \
  --condition $condition --out-dir $RESULT_ROOT --run-name $run; \
echo TRAINING_OK_STARTING_EVAL; \
env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u -m reasoning_loop.boolean_dag_macrostep_eval \
  --checkpoint $RESULT_ROOT/$run/final.pt --out-dir $RESULT_ROOT/$run/eval \
  --batch-size 510 --batches 8; \
echo PIPELINE_COMPLETE"
  tmux new-session -d -s "$session" -c "$ROOT" "bash -lc '$command'"
}

for condition in conditioned no_instruction; do
  while true; do
    if gpu=$(find_shared_gpu); then
      echo "$(date -Is) launching $condition on GPU $gpu"
      launch_condition "$condition" "$gpu"
      sleep 10
      break
    fi
    echo "$(date -Is) waiting for a free GPU for $condition"
    sleep 60
  done
done

echo "$(date -Is) both macro-step pipelines launched"
