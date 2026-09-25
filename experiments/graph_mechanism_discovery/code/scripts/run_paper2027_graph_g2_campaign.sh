#!/usr/bin/env bash
set -euo pipefail
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_graph_g2_evaluation.sh"
seeds="${PAPER2027_GRAPH_SEEDS:-100 101 102 103 104 105 106 107 108 109 110 111}"
gpus="${PAPER2027_GRAPH_GPUS:-0 1 2}"
read -r -a seed_array <<< "$seeds"; read -r -a gpu_array <<< "$gpus"
if (( ${#gpu_array[@]} != 3 )); then echo "PAPER2027_GRAPH_GPUS must name exactly three physical GPUs" >&2; exit 2; fi
for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for slot in 0 1 2; do
    index=$((start + slot)); (( index < ${#seed_array[@]} )) || break
    CUDA_VISIBLE_DEVICES="${gpu_array[slot]}" PAPER2027_GRAPH_SEED="${seed_array[index]}" bash "$runner" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
