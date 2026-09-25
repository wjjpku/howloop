#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 SEED0 SEED1 SEED2 SEED3 SEED4 SEED5 [MORE_SEEDS...]" >&2
  exit 2
fi

readonly candidate_seeds=("$@")
readonly project_dir="/data/wujiaju/LooPlus"
readonly checkpoint_root="/data/wujiaju/graph_path_N10_D8L8_multiseed_20260809/checkpoints"
readonly out_dir="/data/wujiaju/graph_path_N10_D8L8_cycle10_diversity_20260809"
readonly log_path="/data/wujiaju/logs/graph_path_N10_D8L8_multiseed_20260809/cycle10_diversity.log"
readonly python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"

mkdir -p "$out_dir"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0

command=(
  "$python_bin" -u -m scripts.plot_graph_prenorm_multiseed_random_path_readout_20260809
  --checkpoint-root "$checkpoint_root"
  --out-dir "$out_dir"
  --node-count 10
  --seeds "${candidate_seeds[@]}"
  --graph-distribution single_cycle
  --select-diverse-panel
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
