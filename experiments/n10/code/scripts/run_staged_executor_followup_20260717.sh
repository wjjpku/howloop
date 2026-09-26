#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX TRAIN_SEED CIRCUIT_SEED [CIRCUIT_SEED ...]" >&2
  exit 2
fi

gpu_index=$1
train_seed=$2
shift 2
repo_root=${GLOBAL_DEPTH_REPO_ROOT:-/data/paperexperiment/LooPlus}
out_root=${GLOBAL_DEPTH_OUT_ROOT:-/data/paperexperiment/global_depth_supervision_20260717}
python_bin=${GLOBAL_DEPTH_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}

cd "$repo_root"
./scripts/run_graph_global_depth_20260717.sh \
  "$gpu_index" sparse_staged "$train_seed"
./scripts/run_graph_global_depth_analysis_20260717.sh \
  "$gpu_index" sparse_staged "$train_seed"

run_args=()
for seed in "$@"; do
  checkpoint="$out_root/variable_unroll/variable_sparse_staged_N16_D8_d128_seed${seed}/final.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing staged checkpoint: $checkpoint" >&2
    exit 1
  fi
  run_args+=(
    --run
    "staged_seed${seed}=${checkpoint}"
  )
done

CUDA_VISIBLE_DEVICES="$gpu_index" \
  PYTHONPATH="$repo_root" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -m reasoning_loop.graph_path_component_circuit \
    "${run_args[@]}" \
    --out-dir "$out_root/component_circuit_staged/gpu${gpu_index}" \
    --max-loops 8 \
    --batch-size 128 \
    --batches 4 \
    --device cuda
