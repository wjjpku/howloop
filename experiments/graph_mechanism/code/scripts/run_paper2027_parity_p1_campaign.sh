#!/usr/bin/env bash
# Schedule P1 seeds in three GPU-isolated waves.  The individual runner owns
# manifests/logs; this launcher merely keeps at most three training jobs live.
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_parity_input_once_backbone.sh"
seeds="${PAPER2027_PARITY_SEEDS:-3 4 5 6 7 8 9 10 11 12 13 14}"

read -r -a seed_array <<< "$seeds"
if (( ${#seed_array[@]} == 0 || ${#seed_array[@]} % 3 != 0 )); then
  echo "PAPER2027_PARITY_SEEDS must contain a nonempty multiple of three seeds" >&2
  exit 2
fi

for (( start=0; start<${#seed_array[@]}; start+=3 )); do
  pids=()
  labels=()
  for gpu in 0 1 2; do
    seed="${seed_array[start + gpu]}"
    echo "LAUNCH seed=$seed gpu=$gpu $(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="$seed" \
      bash "$runner" &
    pids+=("$!")
    labels+=("seed=$seed,gpu=$gpu")
  done
  failed=0
  for index in 0 1 2; do
    if ! wait "${pids[index]}"; then
      echo "FAILED ${labels[index]} $(date --iso-8601=seconds)" >&2
      failed=1
    else
      echo "COMPLETE ${labels[index]} $(date --iso-8601=seconds)"
    fi
  done
  if (( failed )); then
    echo "stopping campaign after failed wave" >&2
    exit 1
  fi
done
