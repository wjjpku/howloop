#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/home/wujiaju/LooPlus_j_closure_20260801
OUT_DIR=/data/wujiaju/graph_path_j_closure_20260801/seed0_direct_full_affine_r480
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python

cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES=6

"$PYTHON_BIN" -m reasoning_loop.graph_path_j_interval_closure \
  --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary "$CODE_DIR/config/phase_final_seed0.json" \
  --adjacent-bank /data/wujiaju/graph_path_shared_stage_j_20260801/focused5_r48_s16/age_specific_j_bank_after_T05_R5.pt \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --seed 826001 \
  --train-rounds 480 \
  --train-batch-size 64 \
  --learning-rate 3e-4 \
  --save-every-rounds 16 \
  --eval-examples 512 \
  --eval-batch-size 64 \
  --cuda-memory-fraction 0.10
