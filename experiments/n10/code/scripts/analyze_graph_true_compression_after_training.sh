#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU LOOPS SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
loops="$2"
shift 2

training_root="/data/wujiaju/graph_path_true_compression_gate_20260725/training"
analysis_root="/data/wujiaju/graph_path_true_compression_gate_20260725/analysis"
log_root="/data/wujiaju/logs/graph_path_true_compression_gate_20260725/analysis"
export CUDA_VISIBLE_DEVICES="${gpu}"
mkdir -p "${analysis_root}/raw" "${analysis_root}/functional" "${log_root}"

for seed in "$@"; do
  name="D8_L${loops}_seed${seed}"
  run_dir="${training_root}/${name}/graphpath_N8_D8_d256_B2_L${loops}_seed${seed}"
  checkpoint="${run_dir}/best.pt"
  while [[ ! -s "${run_dir}/summary.json" || ! -s "${checkpoint}" ]]; do
    sleep 30
  done

  if [[ ! -s "${analysis_root}/raw/${name}/summary.json" ]]; then
    /data/wujiaju/.venvs/loopreasoner/bin/python -u \
      -m reasoning_loop.graph_path_compression_circuit \
      --run "${name}=${checkpoint}" \
      --out-dir "${analysis_root}/raw" \
      --batch-size 256 \
      --batches 2 \
      --extra-loops 4 \
      --device cuda \
      >"${log_root}/${name}_compression.log" 2>&1
  fi

  if [[ ! -s "${analysis_root}/functional/${name}/summary.json" ]]; then
    /data/wujiaju/.venvs/loopreasoner/bin/python -u \
      -m reasoning_loop.graph_path_functional_circuit \
      --run "${name}=${checkpoint}" \
      --out-dir "${analysis_root}/functional" \
      --batch-size 256 \
      --seed 20260725 \
      --top-k 64 \
      --device cuda \
      --force \
      >"${log_root}/${name}_functional.log" 2>&1
  fi
  echo "ANALYZED ${name}"
done
