#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
physical_gpu="${TRACE_INTERVENTION_GPU:-6}"
evaluation_seed="${TRACE_EVAL_SEED:-820001}"
root=/data/paperexperiment/graph_path_fixed_h1_j_20260801/trace_interventions_20260802
artifact_root="$root/artifacts"
log_root=/data/paperexperiment/logs/graph_path_fixed_h1_j_20260801

case "$mode" in
  smoke)
    trajectories=4
    out_dir="$root/smoke_eval"
    artifacts=(
      /data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt
      "$artifact_root/d_mean.pt"
      "$artifact_root/d_shuffle_seed830001.pt"
      "$artifact_root/d_identity.pt"
      "$artifact_root/ab_rotate_seed830001.pt"
      "$artifact_root/ab_zero.pt"
    )
    ;;
  full)
    trajectories=112
    if [[ "$evaluation_seed" == "820001" ]]; then
      out_dir="$root/full_eval_details"
    else
      out_dir="$root/full_eval_seed${evaluation_seed}"
    fi
    artifacts=(
      /data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt
      "$artifact_root/d_mean.pt"
      "$artifact_root/d_identity.pt"
      "$artifact_root/ab_zero.pt"
      "$artifact_root/d_shuffle_seed830001.pt"
      "$artifact_root/d_shuffle_seed830002.pt"
      "$artifact_root/d_shuffle_seed830003.pt"
      "$artifact_root/d_shuffle_seed830004.pt"
      "$artifact_root/d_shuffle_seed830005.pt"
      "$artifact_root/ab_rotate_seed830001.pt"
      "$artifact_root/ab_rotate_seed830002.pt"
      "$artifact_root/ab_rotate_seed830003.pt"
      "$artifact_root/ab_rotate_seed830004.pt"
      "$artifact_root/ab_rotate_seed830005.pt"
    )
    ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 2
    ;;
esac

mkdir -p "$out_dir" "$log_root"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONPATH=/data/paperexperiment/LooPlus
export HF_HOME=/data/paperexperiment/cache/huggingface
export TRANSFORMERS_CACHE=/data/paperexperiment/cache/huggingface
export TORCH_HOME=/data/paperexperiment/cache/torch

cd /data/paperexperiment/LooPlus

/data/paperexperiment/.venvs/loopreasoner/bin/python \
  reasoning_loop/audit_graph_path_age_specific_j_checkpoint_accuracy.py \
  --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
  --phase-summary /data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json \
  --bank-artifacts "${artifacts[@]}" \
  --out-dir "$out_dir" \
  --device cuda \
  --seed "$evaluation_seed" \
  --trajectories "$trajectories" \
  --batch-size 64 \
  --fixed-start-age 1 \
  --cuda-memory-fraction 0.02 \
  --write-trajectory-details
