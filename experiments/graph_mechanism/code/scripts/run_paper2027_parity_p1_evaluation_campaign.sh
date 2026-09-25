#!/usr/bin/env bash
# Endpoint evaluations run on all P1 seeds; the three prespecified deep seeds
# additionally receive a heatmap, prospective P2 boundary selection, and P3
# phase-causal analysis.
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_parity_p1_evaluation.sh"
endpoint_seeds="${PAPER2027_PARITY_ENDPOINT_SEEDS:-3 4 5 6 7 8 9 10 11 12 13 14}"
deep_seeds="${PAPER2027_PARITY_DEEP_SEEDS:-3 4 5}"

run_wave() {
  local mode="$1"; shift
  local seeds=("$@")
  for ((start=0; start<${#seeds[@]}; start+=3)); do
    pids=()
    for gpu in 0 1 2; do
      local index=$((start + gpu))
      (( index < ${#seeds[@]} )) || break
      CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="${seeds[index]}" \
        PAPER2027_PARITY_EVALUATION_MODE="$mode" bash "$runner" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
}

read -r -a endpoint_array <<< "$endpoint_seeds"
read -r -a deep_array <<< "$deep_seeds"
run_wave deep "${deep_array[@]}"

# Deep runs already include the endpoint horizon, so do not overwrite their
# data with a second nominally identical evaluation.
remaining_endpoint=()
for seed in "${endpoint_array[@]}"; do
  is_deep=false
  for deep_seed in "${deep_array[@]}"; do
    if [[ "$seed" == "$deep_seed" ]]; then is_deep=true; break; fi
  done
  if [[ "$is_deep" == false ]]; then remaining_endpoint+=("$seed"); fi
done
if (( ${#remaining_endpoint[@]} )); then
  run_wave endpoint "${remaining_endpoint[@]}"
fi
