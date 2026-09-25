#!/usr/bin/env bash
set -euo pipefail

cd /data/wujiaju/LooPlus
export CUDA_VISIBLE_DEVICES=2
export TELOMERE_CUDA_MEMORY_FRACTION=0.12

exec /data/wujiaju/.venvs/loopreasoner/bin/python -u \
  -m reasoning_loop.analyze_graph_path_j_late_residual_transplant \
  --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --bank-artifact /data/wujiaju/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt \
  --out-dir /data/wujiaju/graph_path_fixed_h1_j_20260802/attention_compressed_circuit/late_residual_transplant_formal_5seed_7path_v1 \
  --device cuda \
  --rank 8 \
  --ridge 0.001 \
  --profile-mode linear \
  --calibration-examples 1024 \
  --profile-validation-examples 1024 \
  --examples 64 \
  --batch-size 64 \
  --graph-seeds 872001 872002 872003 872004 872005 \
  --back-counts 24 32 48 \
  --path-pairs-per-k 7 \
  --word-seed 872701 \
  --rollback-checkpoints 8 16 24 32 48 \
  --random-draws 2 \
  --strengths 0.5 1.0 \
  --conditions \
    late_bottom_excess_clean0p5 \
    late_bottom_excess_clean1p0 \
    young_bottom_excess_inject0p5 \
    young_bottom_excess_inject1p0 \
    late_random0_excess_clean1p0_statematch \
    late_random1_excess_clean1p0_statematch \
    young_random0_excess_inject1p0_statematch \
    young_random1_excess_inject1p0_statematch \
    late_top_excess_clean1p0_statematch
