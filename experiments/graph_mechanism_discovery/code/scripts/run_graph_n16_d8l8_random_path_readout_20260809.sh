#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 SEED0 SEED1 SEED2 SEED3 SEED4 SEED5" >&2
  exit 2
fi

readonly selected_seeds=("$@")

readonly project_dir="/data/wujiaju/LooPlus"
readonly checkpoint_root="/data/wujiaju/graph_path_N16_D8L8_multiseed_20260809/checkpoints"
readonly out_dir="/data/wujiaju/graph_path_N16_D8L8_multiseed_20260809/random_path_readout"
readonly log_path="/data/wujiaju/logs/graph_path_N16_D8L8_multiseed_20260809/random_path_readout.log"
readonly python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"

mkdir -p "$out_dir"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0

command=(
  "$python_bin" -u scripts/plot_graph_prenorm_multiseed_random_path_readout_20260809.py
  --checkpoint-root "$checkpoint_root"
  --out-dir "$out_dir"
  --node-count 16
  --seeds "${selected_seeds[@]}"
  --device cuda
  --batch-size 512
  --batches 8
  --loops 16
  --data-seed 8091709
)

{
  printf 'physical_gpu=0\n'
  printf 'launch_command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$log_path"

exec "${command[@]}" >>"$log_path" 2>&1
