#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU_INDEX NODE_COUNT DEPTH D_MODEL SEED" >&2
  exit 2
fi

gpu_index=$1
node_count=$2
depth=$3
d_model=$4
seed=$5

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/paperexperiment/LooPlus}
out_root=${RESOURCE_REUSE_OUT_ROOT:-/data/paperexperiment/resource_conditioned_reuse_20260716/graph_shortage_search}
log_root=${RESOURCE_REUSE_LOG_ROOT:-/data/paperexperiment/logs/resource_conditioned_reuse_20260716}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}
steps=${GRAPH_SHORTAGE_STEPS:-12000}
batch_size=${GRAPH_SHORTAGE_BATCH_SIZE:-256}
eval_batch_size=${GRAPH_SHORTAGE_EVAL_BATCH_SIZE:-1024}
eval_batches=${GRAPH_SHORTAGE_EVAL_BATCHES:-16}
d_mlp=$((4 * d_model))
common_name="N${node_count}_D${depth}_d${d_model}_seed${seed}"

mkdir -p "$out_root" "$log_root"
cd "$repo_root"

run_fixed_target() {
  local active_macro_steps=$1
  local arm=$2
  local run_name="${arm}_${common_name}"
  echo "training $run_name on physical GPU $gpu_index"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" -m reasoning_loop.graph_path_fixed_target \
      --node-count "$node_count" \
      --target-depth "$depth" \
      --active-macro-steps "$active_macro_steps" \
      --architecture looped \
      --d-model "$d_model" \
      --n-heads 4 \
      --d-mlp "$d_mlp" \
      --n-layers 2 \
      --steps "$steps" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --eval-batches "$eval_batches" \
      --eval-every 1000 \
      --seed "$seed" \
      --device cuda \
      --run-name "$run_name" \
      --out-dir "$out_root/$arm" \
      >"$log_root/${run_name}.log" 2>&1
}

run_fixed_target 1 "one_step"
run_fixed_target "$depth" "recurrent_final"

transition_name="recurrent_transition_${common_name}"
echo "training $transition_name on physical GPU $gpu_index"
CUDA_VISIBLE_DEVICES="$gpu_index" \
  PYTHONPATH="$repo_root" \
  "$python_bin" -m reasoning_loop.graph_path_stepwise \
    --node-count "$node_count" \
    --max-depth "$depth" \
    --d-model "$d_model" \
    --n-heads 4 \
    --d-mlp "$d_mlp" \
    --n-layers 2 \
    --architecture looped \
    --loops "$depth" \
    --loss-mode transition \
    --steps "$steps" \
    --batch-size "$batch_size" \
    --eval-batch-size "$eval_batch_size" \
    --eval-batches "$eval_batches" \
    --eval-every 1000 \
    --print-every 1000 \
    --seed "$seed" \
    --device cuda \
    --out-dir "$out_root/$transition_name" \
    >"$log_root/${transition_name}.log" 2>&1
