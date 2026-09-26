#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${TRACE_INTERVENTION_GPU:-0}"
evaluation_seed="${TRACE_EVAL_SEED:-883001}"
root=/data/paperexperiment/graph_path_fixed_h1_j_20260802/attention_compressed_circuit/stage24_trace_interventions_v1
artifact_root="$root/artifacts"
out_dir="$root/full_eval"
bank=/data/paperexperiment/graph_path_fixed_h1_j_20260802/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt

mkdir -p "$artifact_root" "$out_dir" /data/paperexperiment/logs
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONPATH=/data/paperexperiment/LooPlus
export HF_HOME=/data/paperexperiment/cache/huggingface
export TRANSFORMERS_CACHE=/data/paperexperiment/cache/huggingface
export TORCH_HOME=/data/paperexperiment/cache/torch

cd /data/paperexperiment/LooPlus

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/make_graph_path_j_trace_interventions.py \
  --bank-artifact "$bank" \
  --out-dir "$artifact_root" \
  --seeds 883101 883102 883103 883104 883105

artifacts=(
  "$bank"
  "$artifact_root/d_mean.pt"
  "$artifact_root/d_identity.pt"
  "$artifact_root/ab_zero.pt"
  "$artifact_root/d_shuffle_seed883101.pt"
  "$artifact_root/d_shuffle_seed883102.pt"
  "$artifact_root/d_shuffle_seed883103.pt"
  "$artifact_root/d_shuffle_seed883104.pt"
  "$artifact_root/d_shuffle_seed883105.pt"
  "$artifact_root/ab_rotate_seed883101.pt"
  "$artifact_root/ab_rotate_seed883102.pt"
  "$artifact_root/ab_rotate_seed883103.pt"
  "$artifact_root/ab_rotate_seed883104.pt"
  "$artifact_root/ab_rotate_seed883105.pt"
)

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/audit_graph_path_age_specific_j_checkpoint_accuracy.py \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary /data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json \
  --bank-artifacts "${artifacts[@]}" \
  --out-dir "$out_dir" \
  --device cuda \
  --seed "$evaluation_seed" \
  --trajectories 112 \
  --batch-size 64 \
  --fixed-start-age 1 \
  --cuda-memory-fraction 0.02 \
  --write-trajectory-details
