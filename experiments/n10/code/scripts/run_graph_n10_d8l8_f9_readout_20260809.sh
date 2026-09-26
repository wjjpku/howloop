#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="/data/paperexperiment/LooPlus"
readonly checkpoint_root="/data/paperexperiment/graph_path_N10_D8L8_multiseed_20260809/checkpoints"
readonly out_dir="/data/paperexperiment/graph_path_N10_D8L8_cycle10_f9_readout_20260809"
readonly log_path="/data/paperexperiment/logs/graph_path_N10_D8L8_multiseed_20260809/cycle10_f9_readout.log"
readonly python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"

mkdir -p "$out_dir"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0

command=(
  "$python_bin" -u -m scripts.plot_graph_prenorm_multiseed_random_path_readout_20260809
  --checkpoint-root "$checkpoint_root"
  --out-dir "$out_dir"
  --node-count 10
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11
  --graph-distribution single_cycle
  --max-path-position 9
  --overview-only
  --plot-both-metrics
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
