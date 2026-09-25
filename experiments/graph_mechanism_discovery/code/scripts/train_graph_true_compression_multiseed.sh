#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU LOOPS SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
loops="$2"
shift 2

if [[ "${loops}" != "3" && "${loops}" != "4" ]]; then
  echo "LOOPS must be 3 or 4" >&2
  exit 2
fi

out_root="/data/wujiaju/graph_path_true_compression_gate_20260725/training"
log_root="/data/wujiaju/logs/graph_path_true_compression_gate_20260725/training"

for seed in "$@"; do
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
