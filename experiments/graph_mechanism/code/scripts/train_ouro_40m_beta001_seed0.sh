#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log_path=/data/wujiaju/logs/ouro_40m_two_task_beta001_seed0_30k.log
output_dir=/data/wujiaju/ouro_mini_runs/stage1_40m_two_task_beta001_seed0_20260713

echo "START $(date -Iseconds)" >"$log_path"
/data/wujiaju/.venvs/loopreasoner/bin/python -u -m ouro_mini.train \
  --model-size 40m \
  --mode stage1 \
  --steps 30000 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --learning-rate 1e-4 \
  --warmup-steps 1000 \
  --entropy-beta-initial 0.01 \
  --entropy-beta-final 0.01 \
  --entropy-switch-step 0 \
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
