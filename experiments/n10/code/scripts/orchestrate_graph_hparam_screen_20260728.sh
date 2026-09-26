#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
root="/data/paperexperiment/graph_path_hparam_circuit_20260728"
repo="/data/paperexperiment/LooPlus"
manifest="${root}/manifests/orchestrator.txt"
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
mkdir -p "$(dirname "${manifest}")" "${log_root}"

status="failed"
finish_manifest() {
  printf 'status=%s\npid=%s\nphysical_gpu=%s\nfinished=%s\nroot=%s\n' \
    "${status}" "$$" "${gpu}" "$(date --iso-8601=seconds)" "${root}" \
    >"${manifest}"
}
trap finish_manifest EXIT
printf 'status=running\npid=%s\nphysical_gpu=%s\nstarted=%s\nroot=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" "${root}" >"${manifest}"

cd "${repo}"
for start in 0 4; do
  pids=()
  for offset in 0 1 2 3; do
    index="$((start + offset))"
    if [[ "${index}" -ge "${#configs[@]}" ]]; then
      continue
    fi
    config="${configs[$index]}"
    ./scripts/train_graph_hparam_config_20260728.sh \
      "${gpu}" "${config}" 0 1 2 3 4 \
      >"${log_root}/launcher_${config}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

for config in "${configs[@]}"; do
  ./scripts/run_graph_hparam_behavior_screen_20260728.sh \
    "${gpu}" "${config}" 0 1 2 3 4
done

status="complete"
