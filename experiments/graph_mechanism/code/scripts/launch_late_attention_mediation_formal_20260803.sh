#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TELOMERE_CUDA_MEMORY_FRACTION="${TELOMERE_CUDA_MEMORY_FRACTION:-0.12}"

exec /data/wujiaju/.venvs/loopreasoner/bin/python -u -m \
  reasoning_loop.analyze_graph_path_j_late_attention_mediation \
  --checkpoint \
  /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --bank-artifact \
  /data/wujiaju/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt \
  --out-dir \
  /data/wujiaju/graph_path_fixed_h1_j_20260802/attention_compressed_circuit/late_attention_mediation_formal_5seed_7path_v1 \
  --device cuda \
  --examples 64 \
  --batch-size 64 \
  --graph-seeds 882001 882002 882003 882004 882005 \
  --back-counts 32 48 \
  --path-pairs-per-k 7 \
  --word-seed 882701 \
  --rollback-checkpoints 24 32 48 \
  --patch-labels \
  B1.H1.qkv \
  B2.H0.q B2.H0.k B2.H0.v B2.H0.qk B2.H0.qkv \
  B2.H0.pattern B2.H0.context \
  B2.H1.qkv \
  B2.attention_out B2.mlp_out
