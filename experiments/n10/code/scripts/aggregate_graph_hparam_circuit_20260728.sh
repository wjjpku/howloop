#!/usr/bin/env bash
set -euo pipefail

root="/data/paperexperiment/graph_path_hparam_circuit_20260728"
repo="/data/paperexperiment/LooPlus"
python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"
configs=(
  baseline_b2
  attn2_mlp05_b2
  block2fast_b2
  beta2_099_b2
  beta1_08_b2
  layers1_b1
  layers3_b3
)
behavior_args=()
functional_args=()
for config in "${configs[@]}"; do
  behavior_args+=(
    --behavior
    "${config}=${root}/behavior_screen/${config}/summary.json"
  )
  functional_args+=(
    --functional
    "${config}=${root}/functional_raw/${config}"
  )
done

cd "${repo}"
PYTHONPATH="${repo}" "${python_bin}" \
  -m reasoning_loop.graph_path_hparam_functional_aggregate \
  "${behavior_args[@]}" \
  "${functional_args[@]}" \
  --baseline baseline_b2 \
  --out-dir "${root}/aggregate"
