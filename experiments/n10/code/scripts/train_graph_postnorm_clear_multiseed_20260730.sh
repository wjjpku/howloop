#!/usr/bin/env bash
set -euo pipefail

code_root=/data/paperexperiment/LooPlus_postnorm_20260730
experiment_root=/data/paperexperiment/graph_path_postnorm_clear_multiseed_20260730
log_root=/data/paperexperiment/logs
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python

mkdir -p "${experiment_root}" "${log_root}"
cd "${code_root}"

for seed in 0 2 3 4 5; do
  condition="trajectory_w1_step14k_seed${seed}"
  output_dir="${experiment_root}/${condition}"
  log_path="${log_root}/graph_path_postnorm_clear_${condition}.log"
  echo "starting ${condition} at $(date --iso-8601=seconds)" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES=2 "${python_bin}" -m reasoning_loop.graph_path_loop \
    --node-count 8 \
    --max-depth 8 \
    --d-model 256 \
    --n-heads 4 \
    --d-mlp 1024 \
    --n-layers 2 \
    --loops 8 \
    --steps 14000 \
    --lr-decay-steps 20000 \
    --batch-size 512 \
    --eval-batch-size 1024 \
    --eval-batches 16 \
    --eval-every 1000 \
    --print-every 1000 \
    --lr 3e-4 \
    --weight-decay 0.3 \
    --warmup-steps 5000 \
    --grad-clip 1.0 \
    --seed "${seed}" \
    --inner-norm-style post_layernorm \
    --aux-loss 0 \
    --trajectory-aux-weight 1.0 \
    --trajectory-aux-jump 2 \
    --no-trajectory-aux-active-only \
    --trajectory-aux-hold-steps 10000 \
    --trajectory-aux-end-step 15000 \
    --device cuda \
    --amp \
    --no-compile \
    --save-checkpoints \
    --no-save-eval-checkpoints \
    --out-dir "${output_dir}" 2>&1 | tee -a "${log_path}"
  echo "completed ${condition} at $(date --iso-8601=seconds)" | tee -a "${log_path}"
done

touch "${experiment_root}/MULTISEED_COMPLETE"
