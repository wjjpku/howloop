#!/usr/bin/env bash
# Exactly the three prespecified deep P1 seeds proceed to P2; no result-driven
# seed selection is permitted.  A no-disease seed is recorded as a no-op.
set -euo pipefail
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_parity_p2_controller.sh"
evaluator="$runner_dir/run_paper2027_parity_p2_evaluation.sh"
seeds="${PAPER2027_PARITY_P2_SEEDS:-3 4 5}"
read -r -a seed_array <<< "$seeds"

for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for gpu in 0 1 2; do
    index=$((start + gpu)); (( index < ${#seed_array[@]} )) || break
    seed="${seed_array[index]}"
    CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="$seed" bash "$runner" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for gpu in 0 1 2; do
    index=$((start + gpu)); (( index < ${#seed_array[@]} )) || break
    seed="${seed_array[index]}"
    CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="$seed" bash "$evaluator" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
