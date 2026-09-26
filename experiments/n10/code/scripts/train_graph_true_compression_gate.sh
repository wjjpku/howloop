#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu="$1"
out_root="/data/paperexperiment/graph_path_true_compression_gate_20260725/training"
log_root="/data/paperexperiment/logs/graph_path_true_compression_gate_20260725/training"

for loops in 4 3; do
  for seed in 0 1; do
    out_dir="${out_root}/D8_L${loops}_seed${seed}"
    log_path="${log_root}/D8_L${loops}_seed${seed}.log"
    checkpoint="${out_dir}/graphpath_N8_D8_d256_B2_L${loops}_seed${seed}/best.pt"
    if [[ -s "${checkpoint}" ]]; then
      echo "SKIP D8_L${loops}_seed${seed}"
      continue
    fi
    "$(dirname "$0")/train_graph_depth_circuit_seed.sh" \
      "${gpu}" 8 "${loops}" "${seed}" "${out_dir}" "${log_path}"
  done
done
