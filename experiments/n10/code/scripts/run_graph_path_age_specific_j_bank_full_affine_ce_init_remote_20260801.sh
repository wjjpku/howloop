#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
source_bank=/data/paperexperiment/graph_path_age_specific_j_bank_20260801/seed0_rank96_final_ce_canonical_init/age_specific_j_bank.pt
out=/data/paperexperiment/graph_path_age_specific_j_bank_20260801/seed0_full_affine_final_ce_rank96_ce_init
log_dir=/data/paperexperiment/logs/graph_path_age_specific_j_bank_20260801
log=${log_dir}/formal_full_affine_ce_init_gpu${gpu}.log

mkdir -p "${out}" "${log_dir}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

"${python_bin}" -m reasoning_loop.train_graph_path_age_specific_j_bank \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --out-dir "${out}" \
    --device cuda \
    --map-architecture full_affine \
    --rank 256 \
    --seed 820001 \
    --initialization age_specific_bank \
    --bank-init-artifact "${source_bank}" \
    --learning-rate-multiplier 0.1 \
    --calibration-heldout-examples 512 \
    --eval-trajectories 56 \
    --eval-batch-size 64 \
    --eval-train-max-backs 28 \
    --eval-unseen-max-backs 40 \
    --composition-examples 256 \
    --composition-batch-size 64 \
    --cuda-memory-fraction 0.045 \
    2>&1 | tee -a "${log}"
