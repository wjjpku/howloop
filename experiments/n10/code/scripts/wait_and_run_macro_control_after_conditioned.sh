#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/paperexperiment/LooPlus
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
RESULT_ROOT=results/boolean_dag_macrostep_seed0_retry2_20260710
LOG=/data/paperexperiment/logs/macro_no_instruction_serial_train.log
GPU=6
EXPECTED_PEAK_MIB=2048
RESERVE_MIB=16384

while tmux has-session -t macro_conditioned_train 2>/dev/null; do
  sleep 60
done

shared_gate_passes() {
  local used total utilization free
  IFS=, read -r used total utilization < <(
    nvidia-smi -i "$GPU" \
      --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
  )
  free=$((total - used))
  (( free >= EXPECTED_PEAK_MIB + RESERVE_MIB && utilization <= 10 ))
}

while true; do
  if shared_gate_passes; then
    sleep 60
    if shared_gate_passes; then
      break
    fi
  fi
  sleep 60
done

cd "$ROOT"
exec > "$LOG" 2>&1
echo "GPU_${GPU}_STARTING_no_instruction_SERIAL_SMOKE"
env CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u -m reasoning_loop.boolean_dag_macrostep_train \
  --condition no_instruction --steps 1 --batch-size 512 --eval-batch-size 4 \
  --eval-batches 1 --eval-every 1 --print-every 1 --out-dir "$RESULT_ROOT" \
  --run-name no_instruction_serial_batch512_smoke
echo "SMOKE_OK_STARTING_20K"
env CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u -m reasoning_loop.boolean_dag_macrostep_train \
  --condition no_instruction --out-dir "$RESULT_ROOT" \
  --run-name no_instruction_seed0_serial
echo "TRAINING_OK_STARTING_EVAL"
env CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u -m reasoning_loop.boolean_dag_macrostep_eval \
  --checkpoint "$RESULT_ROOT/no_instruction_seed0_serial/final.pt" \
  --out-dir "$RESULT_ROOT/no_instruction_seed0_serial/eval" \
  --batch-size 510 --batches 8
echo "PIPELINE_COMPLETE"
