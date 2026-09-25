#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
root="/data/wujiaju/graph_path_induction_contrast_20260726"
repo="/data/wujiaju/LooPlus"
python="/data/wujiaju/.venvs/loopreasoner/bin/python"
manifest="${root}/orchestration_manifest.txt"
log_root="/data/wujiaju/logs/graph_path_induction_contrast_20260726"
mkdir -p "${root}" "${log_root}"

write_status() {
  local status="$1"
  printf 'status=%s\npid=%s\nphysical_gpu=%s\nupdated=%s\n' \
    "${status}" "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"
}

trap 'write_status failed' ERR
write_status running

wait_for_manifests() {
  local prefix="$1"
  while true; do
    local complete=1
    for seed in 0 1 2 3; do
      local path="${root}/${prefix}_seed${seed}.txt"
      if [[ ! -s "${path}" ]] \
        || ! grep -q '^status=complete$' "${path}"; then
        complete=0
        break
      fi
    done
    if [[ "${complete}" -eq 1 ]]; then
      return
    fi
    sleep 30
  done
}

check_checkpoint() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "missing checkpoint: ${path}" >&2
    return 1
  fi
}

cd "${repo}"

echo "WAIT D8 paired training $(date --iso-8601=seconds)"
wait_for_manifests training_manifest
for seed in 0 1 2 3; do
  for loops in 6 8; do
    check_checkpoint \
      "${root}/training/D8_L${loops}_pairseed${seed}/graphpath_N8_D8_d256_B2_L${loops}_seed${seed}/final.pt"
  done
done

echo "START broad 32-checkpoint induction analysis $(date --iso-8601=seconds)"
bash scripts/run_graph_induction_contrast_multiseed_20260726.sh "${gpu}"
echo "COMPLETE broad 32-checkpoint induction analysis $(date --iso-8601=seconds)"

echo "START D8 paired analyses $(date --iso-8601=seconds)"
bash scripts/run_graph_matched_pair_analyses_20260726.sh "${gpu}" 0 1 2 3
"${python}" -m reasoning_loop.graph_path_functional_circuit_aggregate \
  --input-dir "${root}/paired_functional_raw" \
  --out-dir "${root}/paired_functional_aggregate"
"${python}" -m reasoning_loop.graph_path_compression_circuit_aggregate \
  --compression-dir "${root}/paired_compression_raw" \
  --functional-dir "${root}/paired_functional_raw" \
  --out-dir "${root}/paired_compression_aggregate" \
  --paired-seeds
"${python}" -m reasoning_loop.graph_path_induction_contrast_aggregate \
  --raw-dir "${root}/paired_induction_raw" \
  --functional-role-csv \
    "${root}/paired_functional_aggregate/functional_site_roles.csv" \
  --out-dir "${root}/paired_induction_aggregate"
echo "COMPLETE D8 paired analyses $(date --iso-8601=seconds)"

echo "START D6 paired stretch training $(date --iso-8601=seconds)"
for seed in 0 1 2 3; do
  session="graph_matched_stretch_seed${seed}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" \
      "cd ${repo} && bash scripts/train_graph_matched_stretch_pairs.sh ${gpu} ${seed}"
  fi
done
wait_for_manifests training_stretch_manifest
for seed in 0 1 2 3; do
  for loops in 6 8; do
    check_checkpoint \
      "${root}/training_stretch/D6_L${loops}_pairseed${seed}/graphpath_N8_D6_d256_B2_L${loops}_seed${seed}/final.pt"
  done
done
echo "COMPLETE D6 paired stretch training $(date --iso-8601=seconds)"

echo "START D6 paired analyses $(date --iso-8601=seconds)"
bash scripts/run_graph_matched_stretch_analyses_20260726.sh \
  "${gpu}" 0 1 2 3
"${python}" -m reasoning_loop.graph_path_functional_circuit_aggregate \
  --input-dir "${root}/paired_stretch_functional_raw" \
  --out-dir "${root}/paired_stretch_functional_aggregate"
"${python}" -m reasoning_loop.graph_path_compression_circuit_aggregate \
  --compression-dir "${root}/paired_stretch_compression_raw" \
  --functional-dir "${root}/paired_stretch_functional_raw" \
  --out-dir "${root}/paired_stretch_compression_aggregate" \
  --paired-seeds
"${python}" -m reasoning_loop.graph_path_induction_contrast_aggregate \
  --raw-dir "${root}/paired_stretch_induction_raw" \
  --functional-role-csv \
    "${root}/paired_stretch_functional_aggregate/functional_site_roles.csv" \
  --out-dir "${root}/paired_stretch_induction_aggregate"
echo "COMPLETE D6 paired analyses $(date --iso-8601=seconds)"

"${python}" -m reasoning_loop.graph_path_induction_contrast_aggregate \
  --raw-dir "${root}/raw" \
  --functional-role-csv "${root}/prior_functional_site_roles.csv" \
  --out-dir "${root}/aggregate_all_families"
"${python}" -m reasoning_loop.graph_path_paired_contrast_summary \
  --compression-induction-run-summary \
    "${root}/paired_induction_aggregate/run_summary.csv" \
  --stretch-induction-run-summary \
    "${root}/paired_stretch_induction_aggregate/run_summary.csv" \
  --compression-component-rows \
    "${root}/paired_compression_aggregate/component_multifunction_rows.csv" \
  --stretch-component-rows \
    "${root}/paired_stretch_compression_aggregate/component_multifunction_rows.csv" \
  --out-dir "${root}/paired_effect_summary"

write_status complete
echo "ALL COMPLETE $(date --iso-8601=seconds)"
