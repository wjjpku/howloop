#!/usr/bin/env bash
set -euo pipefail

code_root=/data/wujiaju/LooPlus_postnorm_20260730
experiment_root=/data/wujiaju/graph_path_postnorm_clear_circuit_20260730
log_root=/data/wujiaju/logs
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
pre_checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
condition=homotopy_finalonly_pre2k_post12k_lr1e4
output_dir="${experiment_root}/${condition}"
log_path="${log_root}/graph_path_postnorm_clear_${condition}.log"

mkdir -p "${experiment_root}" "${log_root}"
cd "${code_root}"
echo "starting ${condition} at $(date --iso-8601=seconds)" | tee "${log_path}"

CUDA_VISIBLE_DEVICES=2 "${python_bin}" -m reasoning_loop.graph_path_loop \
  --node-count 8 \
  --max-depth 8 \
  --d-model 256 \
  --n-heads 4 \
  --d-mlp 1024 \
  --n-layers 2 \
  --loops 8 \
  --steps 20000 \
  --batch-size 512 \
  --eval-batch-size 1024 \
  --eval-batches 16 \
  --eval-every 1000 \
  --print-every 1000 \
  --lr 1e-4 \
  --weight-decay 0.3 \
  --warmup-steps 5000 \
  --grad-clip 1.0 \
  --seed 1 \
  --init-checkpoint "${pre_checkpoint}" \
  --inner-norm-style post_layernorm \
  --norm-transition-start-step 2000 \
  --norm-transition-end-step 12000 \
  --aux-loss 0 \
  --trajectory-aux-weight 0 \
  --device cuda \
  --amp \
  --no-compile \
  --save-checkpoints \
  --save-eval-checkpoints \
  --out-dir "${output_dir}" 2>&1 | tee -a "${log_path}"

echo "completed ${condition} at $(date --iso-8601=seconds)" | tee -a "${log_path}"
touch "${experiment_root}/HOMOTOPY_COMPLETE"
