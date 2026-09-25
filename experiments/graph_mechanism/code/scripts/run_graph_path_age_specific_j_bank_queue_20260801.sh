#!/usr/bin/env bash
set -uo pipefail

python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
out=/data/wujiaju/graph_path_age_specific_j_bank_20260801/seed0_rank48_final_ce_canonical_init
log_dir=/data/wujiaju/logs/graph_path_age_specific_j_bank_20260801
queue_log=${log_dir}/queue.log
train_log=${log_dir}/train.log
required_free_mib=24576

mkdir -p "${out}" "${log_dir}"

choose_gpu() {
    local gpu free util pids
    for gpu in 0 1 2 3 4 5 6 7; do
        free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        util="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        if [[ "${free}" -ge "${required_free_mib}" && "${util}" -le 10 && -z "${pids}" ]]; then
            echo "${gpu}"
            return 0
        fi
    done
    return 1
}

selected=""
consecutive=0
while [[ "${consecutive}" -lt 2 ]]; do
    candidate="$(choose_gpu || true)"
    printf '%s candidate=%s consecutive=%s\n' "$(date -Is)" "${candidate:-none}" "${consecutive}" >> "${queue_log}"
    if [[ -n "${candidate}" && ( -z "${selected}" || "${candidate}" == "${selected}" ) ]]; then
        selected="${candidate}"
        consecutive=$((consecutive + 1))
    else
        selected="${candidate}"
        consecutive=0
    fi
    [[ "${consecutive}" -lt 2 ]] && sleep 60
done

printf '%s selected_gpu=%s required_free_mib=%s\n' "$(date -Is)" "${selected}" "${required_free_mib}" >> "${queue_log}"
export CUDA_VISIBLE_DEVICES="${selected}"
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
    --canonical-artifact /data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/controllers/final_seed0_ce_h64/task_lora_j.pt \
    --canonical-label task_diagonal_lora_r48_seed211001 \
    --eval-trajectories 56 \
    --eval-batch-size 64 \
    --eval-train-max-backs 28 \
    --eval-unseen-max-backs 40 \
    --composition-examples 256 \
    --composition-batch-size 64 \
    --cuda-memory-fraction 0.06 \
    2>&1 | tee "${train_log}"
status=${PIPESTATUS[0]}
printf '%s exit_status=%s\n' "$(date -Is)" "${status}" >> "${queue_log}"
exit "${status}"
