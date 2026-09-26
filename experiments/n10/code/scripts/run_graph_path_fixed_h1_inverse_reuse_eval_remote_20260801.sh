#!/usr/bin/env bash
set -euo pipefail

cd /data/paperexperiment/LooPlus
export PYTHONPATH=/data/paperexperiment/LooPlus
out_dir=/data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_matched_eval
mkdir -p "$out_dir"

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/audit_graph_path_age_specific_j_checkpoint_accuracy.py \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary /data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json \
  --bank-artifacts \
    /data/paperexperiment/graph_path_fixed_h1_j_20260801/wsd_identity_2x/r64_s16/focused5_h1/age_specific_j_bank_after_T05_R5.pt \
    /data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt \
  --out-dir "$out_dir" \
  --device cuda \
  --seed 820001 \
  --trajectories 112 \
  --batch-size 64 \
  --fixed-start-age 1 \
  --cuda-memory-fraction 0.02

