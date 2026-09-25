#!/usr/bin/env bash
set -euo pipefail

code_root=/data/wujiaju/LooPlus_postnorm_20260730
experiment_root=/data/wujiaju/graph_path_postnorm_clear_circuit_20260730
log_root=/data/wujiaju/logs
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python

mkdir -p "${experiment_root}" "${log_root}"
cd "${code_root}"

common_args=(
  -m reasoning_loop.graph_path_loop
  --node-count 8
  --max-depth 8
  --d-model 256
  --n-heads 4
  --d-mlp 1024
  --n-layers 2
  --loops 8
  --steps 20000
  --batch-size 512
  --eval-batch-size 1024
  --eval-batches 16
  --eval-every 1000
  --print-every 1000
  --lr 3e-4
  --weight-decay 0.3
  --warmup-steps 5000
  --grad-clip 1.0
  --seed 1
  --inner-norm-style post_layernorm
  --aux-loss 0
  --device cuda
  --amp
  --no-compile
  --save-checkpoints
  --save-eval-checkpoints
)

run_condition() {
  local condition=$1
  shift
  local output_dir="${experiment_root}/${condition}"
  local log_path="${log_root}/graph_path_postnorm_clear_${condition}.log"
  echo "starting ${condition} at $(date --iso-8601=seconds)" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES=2 "${python_bin}" "${common_args[@]}" \
    --out-dir "${output_dir}" "$@" 2>&1 | tee -a "${log_path}"
  echo "completed ${condition} at $(date --iso-8601=seconds)" | tee -a "${log_path}"
}

run_condition trajectory_w0p3_hold5k_end10k \
  --trajectory-aux-weight 0.3 \
  --trajectory-aux-jump 2 \
  --trajectory-aux-hold-steps 5000 \
  --trajectory-aux-end-step 10000

run_condition final_only_warmup5k \
  --trajectory-aux-weight 0

touch "${experiment_root}/PIPELINE_COMPLETE"
