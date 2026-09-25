#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU OUT_DIR LOG_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

gpu="$1"
out_dir="$2"
log_dir="$3"
shift 3

mkdir -p "${out_dir}" "${log_dir}"
export CUDA_VISIBLE_DEVICES="${gpu}"

for run_spec in "$@"; do
  name="${run_spec%%=*}"
  checkpoint="${run_spec#*=}"
  if [[ "${name}" == "${checkpoint}" ]]; then
    echo "invalid run spec: ${run_spec}" >&2
    exit 2
  fi
  summary="${out_dir}/${name}/summary.json"
  if [[ -s "${summary}" ]]; then
    echo "SKIP ${name}"
    continue
  fi
  /data/wujiaju/.venvs/loopreasoner/bin/python -u \
    -m reasoning_loop.graph_path_compression_circuit \
    --run "${name}=${checkpoint}" \
    --out-dir "${out_dir}" \
    --batch-size 256 \
    --batches 2 \
    --extra-loops 4 \
    --device cuda \
    >"${log_dir}/${name}.log" 2>&1
  echo "COMPLETE ${name}"
done
