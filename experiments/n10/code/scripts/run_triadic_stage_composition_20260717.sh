#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU_INDEX l1|shared2|unshared2 SEED WIDTH [WIDTH ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
seed=$3
shift 3
case "$arm" in
  l1)
    architecture=looped
    configured_loops=3
    train_loops=1
    ;;
  shared2)
    architecture=looped
    configured_loops=3
    train_loops=2
    ;;
  unshared2)
    architecture=unshared
    configured_loops=2
    train_loops=2
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/paperexperiment/LooPlus}
out_root=${STAGE_COMPOSITION_OUT_ROOT:-/data/paperexperiment/triadic_stage_composition_20260717}
log_root=${STAGE_COMPOSITION_LOG_ROOT:-/data/paperexperiment/logs/triadic_stage_composition_20260717}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}
steps=${STAGE_COMPOSITION_STEPS:-15000}
batch_size=${STAGE_COMPOSITION_BATCH_SIZE:-256}
eval_batch_size=${STAGE_COMPOSITION_EVAL_BATCH_SIZE:-1024}

mkdir -p "$out_root" "$log_root"
cd "$repo_root"

for width in "$@"; do
  run_name="stage_${arm}_p17_d${width}_seed${seed}"
  echo "training $run_name on physical GPU $gpu_index"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" -m reasoning_loop.triadic_stage_composition \
      --p 17 \
      --architecture "$architecture" \
      --d-model "$width" \
      --n-heads 4 \
      --d-mlp "$((2 * width))" \
      --configured-loops "$configured_loops" \
      --train-loops "$train_loops" \
      --steps "$steps" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --diagnostic-batch-size 4096 \
      --eval-every 250 \
      --print-every 250 \
      --seed "$seed" \
      --device cuda \
      --run-name "$run_name" \
      --out-dir "$out_root/$arm" \
      >"$log_root/${run_name}.log" 2>&1
  if (( train_loops >= 2 )); then
    echo "analyzing circuit for $run_name on physical GPU $gpu_index"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.triadic_stage_circuit \
        --checkpoint "$out_root/$arm/$run_name/final.pt" \
        --out-dir "$out_root/analysis/$run_name" \
        --sample-size 1024 \
        --device cuda \
        >>"$log_root/${run_name}.log" 2>&1
  fi
done
