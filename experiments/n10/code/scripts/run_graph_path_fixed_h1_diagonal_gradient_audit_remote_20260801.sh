#!/usr/bin/env bash
set -euo pipefail

cd /data/wujiaju/LooPlus
export PYTHONPATH=/data/wujiaju/LooPlus
out_dir=/data/wujiaju/graph_path_fixed_h1_j_20260801/diagonal_gradient_audit_r48_s16
mkdir -p "$out_dir"

/data/wujiaju/.venvs/loopreasoner/bin/python \
  reasoning_loop/audit_graph_path_fixed_h1_diagonal_gradients.py \
  --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary /data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json \
  --bank-root /data/wujiaju/graph_path_fixed_h1_j_20260801/pure_identity/r48_s16/focused5_h1 \
  --out-dir "$out_dir" \
  --device cuda \
  --batches 28 \
  --batch-size 64 \
  --seed 932001 \
  --cuda-memory-fraction 0.02
