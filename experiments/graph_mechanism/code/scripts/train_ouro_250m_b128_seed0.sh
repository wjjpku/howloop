#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log_path=/data/wujiaju/logs/ouro_250m_two_task_b128_seed0_30k.log
output_dir=/data/wujiaju/ouro_mini_runs/stage1_250m_two_task_b128_seed0_20260712

echo "START $(date -Iseconds)" >"$log_path"
/data/wujiaju/.venvs/loopreasoner/bin/python -u -m ouro_mini.train \
  --model-size 250m \
  --mode stage1 \
  --steps 30000 \
  --batch-size 128 \
  --grad-accum-steps 1 \
  --learning-rate 1e-4 \
  --warmup-steps 1000 \
  --eval-every 1000 \
  --eval-examples-per-cell 64 \
  --save-every 10000 \
  --log-every 100 \
  --seed 0 \
  --device cuda \
  --output-dir "$output_dir" \
  --task-names graph arithmetic \
  >>"$log_path" 2>&1
echo "COMPLETE $(date -Iseconds)" >>"$log_path"
