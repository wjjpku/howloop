#!/usr/bin/env bash
set -euo pipefail

cd /data/paperexperiment/LooPlus
export CUDA_VISIBLE_DEVICES=1
export TELOMERE_CUDA_MEMORY_FRACTION=0.12

exec /data/paperexperiment/.venvs/loopreasoner/bin/python -u \
  -m reasoning_loop.analyze_graph_path_j_stagewise_anti_compression \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --bank-artifact /data/paperexperiment/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt \
  --probe-artifact /data/paperexperiment/graph_path_fixed_h1_j_20260802/compressed_subspace_circuit_stage24_v2/probe_weights_and_bases.npz \
  --out-dir /data/paperexperiment/graph_path_fixed_h1_j_20260802/attention_compressed_circuit/stagewise_anti_compression_formal_5seed_v1 \
  --device cuda \
  --examples 512 \
  --batch-size 64 \
  --graph-seeds 874001 874002 874003 874004 874005 \
  --ranks 4 8 \
  --floors 0.1 0.25 0.5 0.75 1.0 \
  --random-draws 2
