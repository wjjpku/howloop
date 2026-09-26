#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
canonical=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/controllers/final_seed0_ce_h64/task_lora_j.pt
out=/data/paperexperiment/graph_path_age_specific_j_bank_20260801/seed0_rank48_final_ce_canonical_init
log_dir=/data/paperexperiment/logs/graph_path_age_specific_j_bank_20260801
log=${log_dir}/formal_gpu${gpu}.log

mkdir -p "${out}" "${log_dir}"
{
    printf '%s gpu=%s declared_peak_gib=1 reserve_gib=16\n' "$(date -Is)" "${gpu}"
    nvidia-smi -i "${gpu}" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
    for pid in ${pids}; do ps -o user=,pid=,etime=,command= -p "${pid}"; done
} >> "${log}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

"${python_bin}" -m reasoning_loop.train_graph_path_age_specific_j_bank \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --out-dir "${out}" \
    --device cuda \
    --rank 48 \
    --seed 820001 \
    --initialization canonical_shared \
    --canonical-artifact "${canonical}" \
    --canonical-label task_diagonal_lora_r48_seed211001 \
    --eval-trajectories 56 \
    --eval-batch-size 64 \
    --eval-train-max-backs 28 \
    --eval-unseen-max-backs 40 \
    --composition-examples 256 \
    --composition-batch-size 64 \
    --cuda-memory-fraction 0.045 \
    2>&1 | tee -a "${log}"
