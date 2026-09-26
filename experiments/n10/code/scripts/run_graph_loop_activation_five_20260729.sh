#!/usr/bin/env bash
set -euo pipefail

repo="/data/paperexperiment/LooPlus"
run_root="/data/paperexperiment/graph_path_loop_activation_five_models_20260729"
out_dir="${run_root}/raw"
log_dir="/data/paperexperiment/logs/graph_path_loop_activation_five_models_20260729"
log_file="${log_dir}/analysis.log"
manifest="${run_root}/run_manifest.txt"
python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"

mkdir -p "${out_dir}" "${log_dir}"

write_manifest() {
  local status="$1"
  {
    printf 'status=%s\n' "${status}"
    printf 'host=%s\n' "$(hostname)"
    printf 'physical_gpu=2\n'
    printf 'cuda_visible_devices=2\n'
    printf 'batch_size=128\n'
    printf 'top_k=64\n'
    printf 'seed=20260729\n'
    printf 'output_dir=%s\n' "${out_dir}"
    printf 'log_file=%s\n' "${log_file}"
    printf 'updated_at=%s\n' "$(date -Iseconds)"
  } >"${manifest}"
}

write_manifest running
trap 'status=$?; if [[ ${status} -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi' EXIT

cd "${repo}"
CUDA_VISIBLE_DEVICES=2 "${python_bin}" -u \
  -m reasoning_loop.graph_path_loop_activation_circuit \
  --run D8_L6_seed6=/data/paperexperiment/graph_path_functional_multiseed_20260725/training/D8_L6_seed6/graphpath_N8_D8_d256_B2_L6_seed6/best.pt \
  --run D8_L8_seed0=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --run D8_L8_seed1=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt \
  --run D8_L8_seed2=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed2/graphpath_N8_D8_d256_B2_L8_seed2/best.pt \
  --run D8_L8_seed5=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed5/graphpath_N8_D8_d256_B2_L8_seed5/best.pt \
  --out-dir "${out_dir}" \
  --batch-size 128 \
  --top-k 64 \
  --seed 20260729 \
  --device cuda \
  --force \
  2>&1 | tee "${log_file}"
