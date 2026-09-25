#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift

training_root="/data/wujiaju/graph_path_compression_circuit_20260725/training"
compression_out="/data/wujiaju/graph_path_compression_circuit_20260725/analysis/raw"
functional_out="/data/wujiaju/graph_path_compression_circuit_20260725/analysis/functional"
log_root="/data/wujiaju/logs/graph_path_compression_circuit_20260725/analysis"
export CUDA_VISIBLE_DEVICES="${gpu}"
mkdir -p "${compression_out}" "${functional_out}" "${log_root}"

for seed in "$@"; do
  name="D8_L8_seed${seed}"
  run_dir="${training_root}/${name}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
  checkpoint="${run_dir}/best.pt"
  while [[ ! -s "${run_dir}/summary.json" || ! -s "${checkpoint}" ]]; do
    sleep 30
  done

  if [[ ! -s "${compression_out}/${name}/summary.json" ]]; then
    /data/wujiaju/.venvs/loopreasoner/bin/python -u \
      -m reasoning_loop.graph_path_compression_circuit \
      --run "${name}=${checkpoint}" \
      --out-dir "${compression_out}" \
      --batch-size 256 \
      --batches 2 \
      --extra-loops 4 \
      --device cuda \
      >"${log_root}/${name}_compression.log" 2>&1
  fi

  if [[ ! -s "${functional_out}/${name}/summary.json" ]]; then
    /data/wujiaju/.venvs/loopreasoner/bin/python -u \
      -m reasoning_loop.graph_path_functional_circuit \
      --run "${name}=${checkpoint}" \
      --out-dir "${functional_out}" \
      --batch-size 256 \
      --seed 20260725 \
      --top-k 64 \
      --device cuda \
      --force \
      >"${log_root}/${name}_functional.log" 2>&1
  fi
  echo "ANALYZED ${name}"
done
