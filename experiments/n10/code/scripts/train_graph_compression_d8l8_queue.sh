#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift

out_root="/data/paperexperiment/graph_path_compression_circuit_20260725/training"
log_root="/data/paperexperiment/logs/graph_path_compression_circuit_20260725/training"

for seed in "$@"; do
  run_out="${out_root}/D8_L8_seed${seed}"
  run_log="${log_root}/D8_L8_seed${seed}.log"
  checkpoint="${run_out}/graphpath_N8_D8_d256_B2_L8_seed${seed}/best.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "SKIP seed=${seed} checkpoint=${checkpoint}"
    continue
  fi
  "$(dirname "$0")/train_graph_depth_circuit_seed.sh" \
    "${gpu}" 8 8 "${seed}" "${run_out}" "${run_log}"
done
