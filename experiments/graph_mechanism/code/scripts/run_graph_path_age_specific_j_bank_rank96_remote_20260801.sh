#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
canonical=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/controllers/final_seed0_rank96_ce_h64/task_lora_j.pt
out=/data/wujiaju/graph_path_age_specific_j_bank_20260801/seed0_rank96_final_ce_canonical_init
log_dir=/data/wujiaju/logs/graph_path_age_specific_j_bank_20260801
log=${log_dir}/formal_rank96_gpu${gpu}.log

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
    --map-architecture diagonal_lora \
    --rank 96 \
    --seed 820001 \
    --initialization canonical_shared \
    --canonical-artifact "${canonical}" \
    --canonical-label task_diagonal_lora_r96_seed211001 \
    --eval-trajectories 56 \
    --eval-batch-size 64 \
    --eval-train-max-backs 28 \
    --eval-unseen-max-backs 40 \
    --composition-examples 256 \
    --composition-batch-size 64 \
    --cuda-memory-fraction 0.045 \
    2>&1 | tee -a "${log}"
