#!/usr/bin/env bash
set -euo pipefail

cd /data/paperexperiment/LooPlus
export PYTHONPATH=/data/paperexperiment/LooPlus

out_dir=/data/paperexperiment/graph_path_fixed_h1_j_20260801/long_j7_schedule_eval
log=/data/paperexperiment/logs/graph_path_fixed_h1_j_20260801_long_j7_schedule_gpu${CUDA_VISIBLE_DEVICES:-unset}.log
mkdir -p "$out_dir" "$(dirname "$log")"

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/evaluate_graph_path_fixed_j7_long_schedules.py \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary /data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json \
  --bank-artifact /data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt \
  --out-dir "$out_dir" \
  --device cuda \
  --seed 826901 \
  --evaluation-seeds 0 1 2 \
  --examples-per-seed 512 \
  --batch-size 128 \
  --max-forwards 32 \
  --prefix-lengths 1 2 4 7 \
  --rollback-source-age 8 \
  --cuda-memory-fraction 0.02 \
  2>&1 | tee "$log"
