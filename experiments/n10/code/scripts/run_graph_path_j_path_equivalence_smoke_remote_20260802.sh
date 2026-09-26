#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set one physical GPU}"

python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
repo_dir=/data/paperexperiment/LooPlus
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase_summary=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
source_bank=/data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt
out_dir=/data/paperexperiment/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_smoke
log=/data/paperexperiment/logs/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_smoke_gpu${CUDA_VISIBLE_DEVICES}.log

mkdir -p "$out_dir" "$(dirname "$log")"
cd "$repo_dir"

"$python_bin" -u -m reasoning_loop.train_graph_path_j_path_equivalence \
  --checkpoint "$checkpoint" \
  --phase-summary "$phase_summary" \
  --bank-init-artifact "$source_bank" \
  --out-dir "$out_dir" \
  --device cuda \
  --stage-step-limit 1 \
  --batch-size 8 \
  --eval-examples 64 \
  --eval-batch-size 64 \
  --eval-pairs 1 \
  --eval-back-counts 2 \
  --candidates-per-age 2 \
  --cuda-memory-fraction 0.02 \
  --seed 828001 \
  >"$log" 2>&1

test -s "$out_dir/summary.json"
test -s "$out_dir/age_specific_j_bank.pt"

