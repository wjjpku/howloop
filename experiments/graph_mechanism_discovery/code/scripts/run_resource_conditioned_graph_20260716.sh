#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX compressor|standard_transition SEED [SEED ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
shift 2
case "$arm" in
  compressor|standard_transition) ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${RESOURCE_REUSE_OUT_ROOT:-/data/wujiaju/resource_conditioned_reuse_20260716/graph_controls}
log_root=${RESOURCE_REUSE_LOG_ROOT:-/data/wujiaju/logs/resource_conditioned_reuse_20260716}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
mkdir -p "$out_root" "$log_root"
cd "$repo_root"

for seed in "$@"; do
  if [[ "$arm" == "compressor" ]]; then
    run_name="fixed_D6_R1_looped_d256_seed${seed}"
    echo "training $run_name on physical GPU $gpu_index"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.graph_path_fixed_target \
        --node-count 8 \
        --target-depth 6 \
        --active-macro-steps 1 \
        --architecture looped \
        --d-model 256 \
        --n-heads 4 \
        --d-mlp 1024 \
        --n-layers 2 \
        --steps 20000 \
        --batch-size 512 \
        --eval-batch-size 2048 \
        --eval-batches 32 \
        --eval-every 1000 \
        --seed "$seed" \
        --device cuda \
        --run-name "$run_name" \
        --out-dir "$out_root/compressor" \
        >"$log_root/${run_name}.log" 2>&1
  else
    run_name="standard_transition_d256_L6_seed${seed}"
    echo "training $run_name on physical GPU $gpu_index"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.graph_path_stepwise \
        --node-count 8 \
        --max-depth 6 \
        --d-model 256 \
        --n-heads 4 \
        --d-mlp 1024 \
        --n-layers 2 \
        --architecture standard \
        --loops 6 \
        --loss-mode transition \
        --steps 20000 \
        --batch-size 512 \
        --eval-batch-size 2048 \
        --eval-batches 32 \
        --eval-every 1000 \
        --print-every 1000 \
        --seed "$seed" \
        --device cuda \
        --out-dir "$out_root/$run_name" \
        >"$log_root/${run_name}.log" 2>&1
  fi
done
