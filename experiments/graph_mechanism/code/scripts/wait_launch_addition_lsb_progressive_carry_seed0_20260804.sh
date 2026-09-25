#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU=6
MAX_OWN_GPUS=2
REQUIRED_FREE_MIB=18000
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/progressive_carry_addition_20260804
OUT_DIR="${RUN_ROOT}/backbones/addition_lsb_variable_n1to10_progressive_carry_nope_seed0"
LOG_DIR=/data/wujiaju/logs/paper_length_telomere_20260731/progressive_carry_addition_20260804
TRAIN_LOG="${LOG_DIR}/seed0_backbone.log"
CODE_DIR=/data/wujiaju/LooPlus

mkdir -p "${LOG_DIR}"

own_gpu_count() {
    local count=0
    local gpu
    local pid
    local owner
    for gpu in {0..7}; do
        while read -r pid; do
            pid="${pid// /}"
            [[ -n "${pid}" ]] || continue
            owner="$(ps -p "${pid}" -o user= 2>/dev/null | tr -d ' ' || true)"
            if [[ "${owner}" == "wujiaju" ]]; then
                count=$((count + 1))
                break
            fi
        done < <(
            nvidia-smi -i "${gpu}" --query-compute-apps=pid \
                --format=csv,noheader,nounits 2>/dev/null || true
        )
    done
    echo "${count}"
}

while true; do
    own_count="$(own_gpu_count)"
    if (( own_count <= MAX_OWN_GPUS )); then
        break
    fi
    echo "$(date -Is) waiting_for_gpu_quota own_physical_gpus=${own_count}"
    sleep 60
done

while true; do
    own_count="$(own_gpu_count)"
    free_first="$(
        nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free \
            --format=csv,noheader,nounits | tr -d ' '
    )"
    if (( own_count <= MAX_OWN_GPUS && free_first >= REQUIRED_FREE_MIB )); then
        echo "$(date -Is) first_gate own_gpus=${own_count} free_mib=${free_first}"
        sleep 60
        own_count="$(own_gpu_count)"
        free_second="$(
            nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free \
                --format=csv,noheader,nounits | tr -d ' '
        )"
        if (( own_count <= MAX_OWN_GPUS && free_second >= REQUIRED_FREE_MIB )); then
            echo "$(date -Is) second_gate own_gpus=${own_count} free_mib=${free_second}"
            break
        fi
    fi
    echo "$(date -Is) launch_gate_wait own_gpus=${own_count} free_mib=${free_first}"
    sleep 60
done

if [[ -e "${OUT_DIR}" ]]; then
    ABORTED_DIR="${OUT_DIR}.aborted_$(date +%Y%m%dT%H%M%S)"
    mv "${OUT_DIR}" "${ABORTED_DIR}"
    echo "$(date -Is) preserved_partial_run=${ABORTED_DIR}"
fi

echo "$(date -Is) launching gpu=${PHYSICAL_GPU} out_dir=${OUT_DIR}"
cd "${CODE_DIR}"
exec env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/run_addition_lsb_progressive_carry_backbone_20260804.sh 0 \
    >> "${TRAIN_LOG}" 2>&1
