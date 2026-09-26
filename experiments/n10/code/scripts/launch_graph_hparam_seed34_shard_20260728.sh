#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
repo="/data/paperexperiment/LooPlus"
log_root="/data/paperexperiment/logs/graph_path_hparam_circuit_20260728"
configs=(
  baseline_b2
  attn2_mlp05_b2
  block2fast_b2
  beta2_099_b2
  beta1_08_b2
  layers1_b1
  layers3_b3
)
cd "${repo}"

pids=()
for config in "${configs[@]}"; do
  GRAPH_HPARAM_MANIFEST_SUFFIX="_seed34" \
    ./scripts/train_graph_hparam_config_20260728.sh \
      "${gpu}" "${config}" 3 4 \
      >"${log_root}/launcher_${config}_seed34.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
