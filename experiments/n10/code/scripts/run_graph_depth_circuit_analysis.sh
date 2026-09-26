#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU OUT_DIR LOG_PATH NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

gpu="$1"
out_dir="$2"
log_path="$3"
shift 3

mkdir -p "${out_dir}" "$(dirname "${log_path}")"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec >"${log_path}" 2>&1

echo "START $(date --iso-8601=seconds)"
run_args=()
for run_spec in "$@"; do
  run_args+=(--run "${run_spec}")
done
/data/paperexperiment/.venvs/loopreasoner/bin/python -u \
  -m reasoning_loop.graph_path_depth_circuit \
  "${run_args[@]}" \
  --out-dir "${out_dir}" \
  --device cuda \
  --batch-size 1024 \
  --batches 2 \
  --overloops 16 \
  --seed 20260725
echo "COMPLETE $(date --iso-8601=seconds)"
