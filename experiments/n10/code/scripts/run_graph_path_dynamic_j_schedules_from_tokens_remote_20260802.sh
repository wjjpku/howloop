#!/usr/bin/env bash
set -euo pipefail

cd /data/paperexperiment/LooPlus
export PYTHONPATH=/data/paperexperiment/LooPlus
out_dir=/data/paperexperiment/graph_path_fixed_h1_j_20260801/dynamic_j_schedules_from_h1
log=/data/paperexperiment/logs/graph_path_dynamic_j_schedules_from_h1_gpu${CUDA_VISIBLE_DEVICES}.log
mkdir -p "$out_dir" "$(dirname "$log")"

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/evaluate_graph_path_dynamic_j_schedules_from_tokens.py \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --bank-artifact /data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt \
  --out-dir "$out_dir" \
  --device cuda \
  --seed 827401 \
  --evaluation-seeds 0 1 2 \
  --examples-per-seed 512 \
  --batch-size 128 \
  --max-state 40 \
  --prefix-lengths 1 2 4 7 \
  --cuda-memory-fraction 0.02 \
  2>&1 | tee "$log"
