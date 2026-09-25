#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

gpu_index=$1
repo_root=${SHARDED_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${SHARDED_REUSE_OUT_ROOT:-/data/wujiaju/sharded_evidence_reuse_20260715}
python_bin=${SHARDED_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
analysis_root="$out_root/analysis"
mkdir -p "$analysis_root"

mapfile -t checkpoints < <(
  find "$out_root" -type f -name checkpoint_step_0001000.pt \
    -not -path '*/analysis/*' | sort
)
if [[ ${#checkpoints[@]} -ne 24 ]]; then
  echo "expected 24 final checkpoints, found ${#checkpoints[@]}" >&2
  printf '%s\n' "${checkpoints[@]}" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  run_name=$(basename "$(dirname "$checkpoint")")
  echo "analyzing $run_name"
  (
    cd "$repo_root"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.sharded_evidence_diagnostics \
        --checkpoint "$checkpoint" \
        --out-dir "$analysis_root/$run_name" \
        --analysis-condition shard \
        --batch-size 512 \
        --batches 8 \
        --device cuda
  ) >"$analysis_root/$run_name.log" 2>&1
done

touch "$out_root/analysis_queue_complete"
echo "completed sharded-evidence analysis queue"
