#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-4}"
repo="/data/paperexperiment/LooPlus"
log_root="/data/paperexperiment/logs/graph_path_hparam_circuit_20260728"
configs=(beta1_08_b2 layers1_b1 layers3_b3)
cd "${repo}"

pids=()
for config in "${configs[@]}"; do
  ./scripts/train_graph_hparam_config_20260728.sh \
    "${gpu}" "${config}" 0 1 2 3 4 \
    >"${log_root}/launcher_${config}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
