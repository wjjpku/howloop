#!/usr/bin/env bash
set -euo pipefail

run_root=/data/wujiaju/parity_input_once_20260811
code_file="$run_root/code/paper_length_telomere.py"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
out_dir="$run_root/backbones/parity_input_once_seed0"
log_file=/data/wujiaju/logs/parity_input_once_20260811/backbone_seed0.log
pid_file="$run_root/backbone_seed0.pid"

mkdir -p "$out_dir" "$(dirname "$log_file")"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1

echo "START $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUTPUT=$out_dir"

"$python_bin" -u "$code_file" backbone \
  --task parity \
  --supervision adaptive_step \
  --official-model-config \
  --token-embedding-injection initial_only \
  --steps 100001 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --seed 0 \
  --device auto \
  --no-amp \
  --log-every 100 \
  --eval-every 1000 \
  --eval-batch-size 256 \
  --eval-batches 4 \
  --checkpoint-every 10000 \
  --out-dir "$out_dir"

echo "COMPLETE $(date --iso-8601=seconds)"
