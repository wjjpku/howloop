#!/usr/bin/env bash
set -euo pipefail

cd /data/wujiaju/LooPlus
export CUDA_VISIBLE_DEVICES=0
export TELOMERE_CUDA_MEMORY_FRACTION=0.12

exec /data/wujiaju/.venvs/loopreasoner/bin/python -u \
  -m reasoning_loop.analyze_graph_path_j_centered_residual_telomere \
  --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --bank-artifact /data/wujiaju/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt \
  --probe-artifact /data/wujiaju/graph_path_fixed_h1_j_20260802/compressed_subspace_circuit_stage24_v2/probe_weights_and_bases.npz \
  --out-dir /data/wujiaju/graph_path_fixed_h1_j_20260802/attention_compressed_circuit/centered_residual_formal_5seed_7path_v1 \
  --device cuda \
  --rank 8 \
  --ridge 0.001 \
  --calibration-examples 1024 \
  --profile-validation-examples 1024 \
  --examples 64 \
  --batch-size 64 \
  --graph-seeds 870001 870002 870003 870004 870005 \
  --back-counts 16 24 32 48 64 \
  --path-pairs-per-k 7 \
  --word-seed 870701 \
  --random-draws 2 \
  --strengths 0.1 0.25 0.5 1.0 \
  --conditions \
    bottom_mean_shrink0p1 \
    bottom_mean_shrink0p25 \
    bottom_mean_shrink0p5 \
    bottom_linear_shrink0p1 \
    bottom_linear_shrink0p25 \
    bottom_linear_shrink0p5 \
    bottom_linear_shrink1p0 \
    bottom_linear_expand0p25 \
    bottom_linear_expand0p5 \
    random0_linear_shrink0p25_statematch \
    random1_linear_shrink0p25_statematch \
    random0_linear_shrink0p5_statematch \
    random1_linear_shrink0p5_statematch \
    random0_linear_expand0p5_statematch \
    random1_linear_expand0p5_statematch \
    top_linear_shrink0p25_statematch \
    top_linear_shrink0p5_statematch
