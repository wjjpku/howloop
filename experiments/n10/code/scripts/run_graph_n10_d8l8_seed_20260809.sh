#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED PHYSICAL_GPU" >&2
  exit 2
fi

readonly seed="$1"
readonly physical_gpu="$2"
readonly project_dir="/data/paperexperiment/LooPlus"
readonly result_root="/data/paperexperiment/graph_path_N10_D8L8_multiseed_20260809/checkpoints"
readonly log_root="/data/paperexperiment/logs/graph_path_N10_D8L8_multiseed_20260809"
readonly python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"
readonly seed_out_dir="${result_root}/D8_L8_seed${seed}"
readonly log_path="${log_root}/train_seed${seed}.log"

mkdir -p "$seed_out_dir" "$log_root"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES="$physical_gpu"

command=(
  "$python_bin" -u -m reasoning_loop.graph_path_loop
  --node-count 10
  --max-depth 8
  --d-model 256
  --n-heads 4
  --d-mlp 1024
  --n-layers 2
  --loops 8
  --steps 20000
  --batch-size 512
  --eval-batch-size 1024
  --eval-batches 16
  --eval-every 1000
  --print-every 1000
  --lr 0.0003
  --weight-decay 0.3
  --warmup-steps 500
  --grad-clip 1.0
  --seed "$seed"
  --dropout 0.0
  --aux-loss 0.0
  --device cuda
  --amp
  --no-compile
  --save-checkpoints
  --no-save-eval-checkpoints
  --out-dir "$seed_out_dir"
)

{
  printf 'physical_gpu=%s\n' "$physical_gpu"
  printf 'launch_command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$log_path"

exec "${command[@]}" >>"$log_path" 2>&1
