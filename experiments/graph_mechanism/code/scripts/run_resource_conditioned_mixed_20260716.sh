#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX nomode|mode SEED [SEED ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
shift 2
case "$arm" in
  nomode) mode_flag=() ;;
  mode) mode_flag=(--mode-conditioning) ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${RESOURCE_REUSE_OUT_ROOT:-/data/wujiaju/resource_conditioned_reuse_20260716/triadic_mixed/runs}
log_root=${RESOURCE_REUSE_LOG_ROOT:-/data/wujiaju/logs/resource_conditioned_reuse_20260716}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
mkdir -p "$out_root" "$log_root"
cd "$repo_root"

for seed in "$@"; do
  run_name="mixed_${arm}_d64_L6_seed${seed}"
  echo "training $run_name on physical GPU $gpu_index"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" -m reasoning_loop.triadic_mixed_schedule_train \
      --steps 10000 \
      --eval-every 250 \
      --checkpoint-every 250 \
      --batch-size 256 \
      --eval-batch-size 1024 \
      --seed "$seed" \
      "${mode_flag[@]}" \
      --run-name "$run_name" \
      --out-dir "$out_root" \
      >"$log_root/${run_name}.log" 2>&1
done
