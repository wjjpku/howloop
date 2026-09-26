#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU CONFIG [CONFIG ...]" >&2
  exit 2
fi

gpu="$1"
shift
repo="/data/paperexperiment/LooPlus"
log_root="/data/paperexperiment/logs/graph_path_hparam_circuit_20260728"
cd "${repo}"

pids=()
for config in "$@"; do
  ./scripts/run_graph_hparam_functional_config_20260728.sh \
    "${gpu}" "${config}" 0 1 2 3 4 \
    >"${log_root}/functional_launcher_${config}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
