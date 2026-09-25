#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
root="/data/wujiaju/graph_path_induction_contrast_d64_20260727"
repo="/data/wujiaju/LooPlus"
python="/data/wujiaju/.venvs/loopreasoner/bin/python"
log_root="/data/wujiaju/logs/graph_path_induction_contrast_d64_20260727/analysis"
manifest="${root}/analysis_manifest.txt"
mkdir -p "${log_root}"
printf 'status=running\npid=%s\nphysical_gpu=%s\nstarted=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${repo}"

run_family() {
  local depth="$1"
  local prefix="$2"
  local top_k=16
  local run_args=()
  for seed in 0 1 2 3; do
    for loops in 6 8; do
      local checkpoint="${root}/training/D${depth}_L${loops}_pairseed${seed}/graphpath_N8_D${depth}_d64_B2_L${loops}_seed${seed}/final.pt"
      if [[ ! -s "${checkpoint}" ]]; then
        echo "missing checkpoint: ${checkpoint}" >&2
        return 1
      fi
      run_args+=(--run "D${depth}_L${loops}_seed${seed}=${checkpoint}")
    done
  done

  {
    echo "START ${prefix} induction $(date --iso-8601=seconds)"
    "${python}" -u -m reasoning_loop.graph_path_induction_contrast \
      "${run_args[@]}" \
      --out-dir "${root}/${prefix}_induction_raw" \
      --batch-size 256 \
      --max-bundle-edges 4 \
      --device cuda \
      --force
    echo "COMPLETE ${prefix} induction $(date --iso-8601=seconds)"
  } >"${log_root}/${prefix}_induction.log" 2>&1

  {
    echo "START ${prefix} functional $(date --iso-8601=seconds)"
    "${python}" -u -m reasoning_loop.graph_path_functional_circuit \
      "${run_args[@]}" \
      --out-dir "${root}/${prefix}_functional_raw" \
      --batch-size 256 \
      --top-k "${top_k}" \
      --device cuda \
      --force
    echo "COMPLETE ${prefix} functional $(date --iso-8601=seconds)"
  } >"${log_root}/${prefix}_functional.log" 2>&1

  {
    echo "START ${prefix} compression $(date --iso-8601=seconds)"
    "${python}" -u -m reasoning_loop.graph_path_compression_circuit \
      "${run_args[@]}" \
      --out-dir "${root}/${prefix}_compression_raw" \
      --batch-size 256 \
      --batches 8 \
      --extra-loops 4 \
      --device cuda
    echo "COMPLETE ${prefix} compression $(date --iso-8601=seconds)"
  } >"${log_root}/${prefix}_compression.log" 2>&1

  "${python}" -m reasoning_loop.graph_path_functional_circuit_aggregate \
    --input-dir "${root}/${prefix}_functional_raw" \
    --out-dir "${root}/${prefix}_functional_aggregate" \
    --top-k "${top_k}"
  "${python}" -m reasoning_loop.graph_path_compression_circuit_aggregate \
    --compression-dir "${root}/${prefix}_compression_raw" \
    --functional-dir "${root}/${prefix}_functional_raw" \
    --out-dir "${root}/${prefix}_compression_aggregate" \
    --top-k "${top_k}" \
    --paired-seeds
  "${python}" -m reasoning_loop.graph_path_induction_contrast_aggregate \
    --raw-dir "${root}/${prefix}_induction_raw" \
    --functional-role-csv \
      "${root}/${prefix}_functional_aggregate/functional_site_roles.csv" \
    --out-dir "${root}/${prefix}_induction_aggregate"
}

run_family 8 compression
run_family 6 stretch

"${python}" -m reasoning_loop.graph_path_paired_contrast_summary \
  --compression-induction-run-summary \
    "${root}/compression_induction_aggregate/run_summary.csv" \
  --stretch-induction-run-summary \
    "${root}/stretch_induction_aggregate/run_summary.csv" \
  --compression-component-rows \
    "${root}/compression_compression_aggregate/component_multifunction_rows.csv" \
  --stretch-component-rows \
    "${root}/stretch_compression_aggregate/component_multifunction_rows.csv" \
  --out-dir "${root}/paired_effect_summary"

printf 'status=complete\npid=%s\nphysical_gpu=%s\ncompleted=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"
