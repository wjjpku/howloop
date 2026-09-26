#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
root=/data/paperexperiment/graph_path_age_specific_j_bank_20260801
log_dir=/data/paperexperiment/logs/graph_path_age_specific_j_bank_20260801

mkdir -p "${log_dir}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

evaluate_bank() {
    local label="$1"
    local bank="$2"
    local out="${root}/${label}"
    local log="${log_dir}/${label}_gpu${gpu}.log"
    test ! -e "${out}/summary.json"
    "${python_bin}" -m reasoning_loop.train_graph_path_age_specific_j_bank \
        --checkpoint "${checkpoint}" \
        --phase-summary "${phase}" \
        --out-dir "${out}" \
        --device cuda \
        --map-architecture full_affine \
        --rank 256 \
        --seed 820001 \
        --initialization age_specific_bank \
        --bank-init-artifact "${bank}" \
        --curriculum focused5 \
        --max-stages 0 \
        --eval-trajectories 56 \
        --eval-batch-size 64 \
        --composition-examples 256 \
        --composition-batch-size 64 \
        --cuda-memory-fraction 0.045 \
        2>&1 | tee "${log}"
}

evaluate_bank \
    seed0_full_affine_single_back_focused5_fixed_eval \
    "${root}/seed0_full_affine_single_back/age_specific_j_bank.pt"

evaluate_bank \
    seed0_full_affine_focused5_fixed_eval \
    "${root}/seed0_full_affine_focused5_curriculum/age_specific_j_bank.pt"
