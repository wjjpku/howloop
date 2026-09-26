#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LABEL BETA_INITIAL BETA_FINAL SWITCH_STEP" >&2
  exit 2
fi

label="$1"
beta_initial="$2"
beta_final="$3"
switch_step="$4"

if [[ ! "$label" =~ ^[a-z0-9_]+$ ]]; then
  echo "label must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one physical GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log_path="/data/paperexperiment/logs/ouro_40m_two_task_${label}_seed0_30k.log"
output_dir="/data/paperexperiment/ouro_mini_runs/stage1_40m_two_task_${label}_seed0_20260713"

echo "START $(date -Iseconds)" >"$log_path"
/data/paperexperiment/.venvs/loopreasoner/bin/python -u -m ouro_mini.train \
  --model-size 40m \
  --mode stage1 \
  --steps 30000 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --learning-rate 1e-4 \
  --warmup-steps 1000 \
  --entropy-beta-initial "$beta_initial" \
  --entropy-beta-final "$beta_final" \
  --entropy-switch-step "$switch_step" \
  --eval-every 1000 \
  --eval-examples-per-cell 64 \
  --save-every 10000 \
  --log-every 100 \
  --seed 0 \
  --device cuda \
  --no-gradient-checkpointing \
  --output-dir "$output_dir" \
  --task-names graph arithmetic \
  >>"$log_path" 2>&1
echo "COMPLETE $(date -Iseconds)" >>"$log_path"
