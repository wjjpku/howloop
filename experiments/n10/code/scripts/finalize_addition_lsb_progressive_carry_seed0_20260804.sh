#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU=6
MAX_OWN_GPUS=2
REQUIRED_FREE_MIB=18000
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/progressive_carry_addition_20260804
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
BACKBONE_DIR="${RUN_ROOT}/backbones/addition_lsb_variable_n1to10_progressive_carry_nope_seed0"
LOG_DIR=/data/paperexperiment/logs/paper_length_telomere_20260731/progressive_carry_addition_20260804

mkdir -p "${LOG_DIR}"
cd "${CODE_DIR}"

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
            if [[ "${owner}" == "researcher" ]]; then
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

wait_for_launch_gate() {
    while true; do
        local own_count
        local free_first
        local free_second
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
                return
            fi
        fi
        echo "$(date -Is) resource_wait own_gpus=${own_count} free_mib=${free_first}"
        sleep 60
    done
}

gate_pass() {
    local csv_path="$1"
    awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "variant") variant_col = i
                if ($i == "supervised_digit_exact_match") metric_col = i
            }
            next
        }
        $variant_col == "raw" && ($metric_col + 0) < 0.995 { failed = 1 }
        END {
            if (!variant_col || !metric_col) exit 2
            exit failed
        }
    ' "${csv_path}"
}

while [[ ! -f "${LOG_DIR}/seed0_queue.exit_code" ]]; do
    echo "$(date -Is) waiting_for_backbone_queue"
    sleep 60
done
if [[ "$(cat "${LOG_DIR}/seed0_queue.exit_code")" != "0" ]]; then
    echo "seed0 backbone queue exited unsuccessfully" >&2
    exit 31
fi
if [[ ! -f "${BACKBONE_DIR}/checkpoint_080000.pt" ]]; then
    echo "seed0 backbone has no 80k checkpoint" >&2
    exit 32
fi

SELECTED_STEP=80000
for STEP in 80000 100000; do
    CHECKPOINT="${BACKBONE_DIR}/checkpoint_$(printf '%06d' "${STEP}").pt"
    GATE_DIR="${RUN_ROOT}/audits/seed0_raw_id_step${STEP}"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        if [[ "${STEP}" == "100000" ]]; then
            wait_for_launch_gate
            CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                bash scripts/continue_addition_lsb_progressive_carry_backbone_20260804.sh \
                0 80000 100000 >> "${LOG_DIR}/seed0_backbone_continue100k.log" 2>&1
        else
            exit 33
        fi
    fi
    wait_for_launch_gate
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
        "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
        --checkpoint "${CHECKPOINT}" \
        --variants raw \
        --lengths 1 2 3 4 5 6 7 8 9 10 \
        --batch-size 128 --batches 8 --seed 585001 --device cuda \
        --out-dir "${GATE_DIR}"
    if gate_pass "${GATE_DIR}/endpoint_accuracy.csv"; then
        SELECTED_STEP="${STEP}"
        break
    fi
    SELECTED_STEP=0
done

if [[ "${SELECTED_STEP}" == "0" ]]; then
    echo "progressive-carry baseline failed the full-arithmetic ID gate" >&2
    exit 34
fi

CHECKPOINT="${BACKBONE_DIR}/checkpoint_$(printf '%06d' "${SELECTED_STEP}").pt"
CONTROLLER_DIR="${RUN_ROOT}/controllers/addition_lsb_progressive_carry_seed0_step${SELECTED_STEP}_rank48_identity_anchor1_wsd5376_controllerseed523101"
wait_for_launch_gate
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/run_addition_lsb_progressive_carry_j_20260804.sh \
    "${CHECKPOINT}" "${CONTROLLER_DIR}" \
    >> "${LOG_DIR}/seed0_controller.log" 2>&1

wait_for_launch_gate
AUDIT_DIR="${RUN_ROOT}/audits/seed0_raw_vs_j_n1to30"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
    "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER_DIR}/controller.pt" \
    --variants raw full no_AB identity_D \
    --lengths {1..30} \
    --batch-size 128 --batches 8 --seed 585001 --device cuda \
    --out-dir "${AUDIT_DIR}"

echo "$(date -Is) progressive_carry_seed0_complete selected_step=${SELECTED_STEP}"
