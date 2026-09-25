#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 SEED BACKBONE_PID PHYSICAL_GPU" >&2
    exit 2
fi

SEED="$1"
BACKBONE_PID="$2"
PHYSICAL_GPU="$3"
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804
CODE_DIR=/data/wujiaju/LooPlus
PYTHON=/data/wujiaju/.venvs/loopreasoner/bin/python
BACKBONE_DIR="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed${SEED}"
REPLICATION_ROOT="${RUN_ROOT}/replication_seed${SEED}"
REQUIRED_FREE_MIB=18000
REQUIRED_DISK_MIB=51200

mkdir -p "${REPLICATION_ROOT}"
cd "${CODE_DIR}"

wait_for_resources() {
    while true; do
        FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        DISK_MIB="$(df --output=avail -BM /data | tail -n 1 | tr -dc '0-9')"
        if (( DISK_MIB < REQUIRED_DISK_MIB )); then
            echo "$(date -Is) abort_low_disk available_mib=${DISK_MIB}" >&2
            exit 51
        fi
        if (( FREE_MIB >= REQUIRED_FREE_MIB )); then
            echo "$(date -Is) resource_gate_pass gpu=${PHYSICAL_GPU} free_mib=${FREE_MIB} disk_mib=${DISK_MIB}"
            return
        fi
        echo "$(date -Is) resource_gate_wait gpu=${PHYSICAL_GPU} free_mib=${FREE_MIB} required_mib=${REQUIRED_FREE_MIB}"
        sleep 60
    done
}

gate_pass() {
    local csv_path="$1"
    local variant="$2"
    awk -F, -v wanted="${variant}" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "variant") variant_col = i
                if ($i == "supervised_digit_exact_match") metric_col = i
            }
            next
        }
        $variant_col == wanted && ($metric_col + 0) < 0.995 { failed = 1 }
        END {
            if (!variant_col || !metric_col) exit 2
            exit failed
        }
    ' "${csv_path}"
}

evaluate_raw_gate() {
    local step="$1"
    local checkpoint
    local out_dir
    checkpoint="${BACKBONE_DIR}/checkpoint_$(printf '%06d' "${step}").pt"
    out_dir="${REPLICATION_ROOT}/baseline_gate_step${step}"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "missing checkpoint for gate: ${checkpoint}" >&2
        return 2
    fi
    if [[ ! -f "${out_dir}/summary.json" ]] || ! grep -q '"status": "complete"' "${out_dir}/summary.json"; then
        wait_for_resources
        CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
            "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
            --checkpoint "${checkpoint}" \
            --variants raw \
            --lengths 1 2 3 4 5 6 7 8 9 10 \
            --batch-size 128 \
            --batches 8 \
            --seed 584001 \
            --device cuda \
            --out-dir "${out_dir}"
    fi
    gate_pass "${out_dir}/endpoint_accuracy.csv" raw
}

while kill -0 "${BACKBONE_PID}" 2>/dev/null; do
    echo "$(date -Is) waiting_backbone seed=${SEED} pid=${BACKBONE_PID}"
    sleep 60
done

if [[ ! -f "${BACKBONE_DIR}/summary.json" ]] \
    || ! grep -q '"status": "complete"' "${BACKBONE_DIR}/summary.json" \
    || [[ ! -f "${BACKBONE_DIR}/checkpoint_080000.pt" ]]; then
    echo "backbone seed ${SEED} exited without complete 80k artifacts" >&2
    exit 31
fi

declare -A GATE
for STEP in 75000 80000; do
    if evaluate_raw_gate "${STEP}"; then
        GATE[${STEP}]=1
    else
        GATE[${STEP}]=0
    fi
    echo "$(date -Is) baseline_gate seed=${SEED} step=${STEP} passed=${GATE[${STEP}]}"
done

SELECTED_STEP=0
if [[ "${GATE[75000]}" -eq 1 && "${GATE[80000]}" -eq 1 ]]; then
    SELECTED_STEP=80000
else
    wait_for_resources
    echo "$(date -Is) baseline_continue seed=${SEED} from=80000 to=100000"
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        bash "${CODE_DIR}/scripts/continue_addition_lsb_variable_tn_backbone_seed_20260804.sh" \
        "${SEED}" 80000 100000
    PREVIOUS_STEP=80000
    for STEP in 85000 90000 95000 100000; do
        if evaluate_raw_gate "${STEP}"; then
            GATE[${STEP}]=1
        else
            GATE[${STEP}]=0
        fi
        echo "$(date -Is) baseline_gate seed=${SEED} step=${STEP} passed=${GATE[${STEP}]}"
        if [[ "${GATE[${PREVIOUS_STEP}]}" -eq 1 && "${GATE[${STEP}]}" -eq 1 ]]; then
            SELECTED_STEP="${STEP}"
            break
        fi
        PREVIOUS_STEP="${STEP}"
    done
fi

if [[ "${SELECTED_STEP}" -eq 0 ]]; then
    echo "seed ${SEED} never passed two consecutive 5k-spaced ID gates" >&2
    exit 32
fi

SELECTED_CHECKPOINT="${BACKBONE_DIR}/checkpoint_$(printf '%06d' "${SELECTED_STEP}").pt"
CONTROLLER_DIR="${RUN_ROOT}/controllers/addition_lsb_variable_m1to10_tn${SELECTED_STEP}_seed${SEED}_rank48_identity_logicaldigits_m2to10_anchor1_wsd5376_controllerseed521101"
CONTROLLER="${CONTROLLER_DIR}/controller.pt"
echo "$(date -Is) selected_backbone seed=${SEED} step=${SELECTED_STEP} checkpoint=${SELECTED_CHECKPOINT}"

if [[ ! -f "${CONTROLLER}" ]]; then
    wait_for_resources
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        bash "${CODE_DIR}/scripts/run_addition_lsb_variable_tn_j_20260804.sh" \
        "${SELECTED_CHECKPOINT}" "${CONTROLLER_DIR}"
fi
if [[ ! -f "${CONTROLLER}" ]]; then
    echo "J training did not produce controller: ${CONTROLLER}" >&2
    exit 33
fi

ID_DIR="${REPLICATION_ROOT}/j_id_l1to10_n1024"
if [[ ! -f "${ID_DIR}/summary.json" ]] || ! grep -q '"status": "complete"' "${ID_DIR}/summary.json"; then
    wait_for_resources
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
        "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
        --checkpoint "${SELECTED_CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants raw full \
        --lengths 1 2 3 4 5 6 7 8 9 10 \
        --batch-size 128 --batches 8 --seed 584001 --device cuda \
        --out-dir "${ID_DIR}"
fi
if ! gate_pass "${ID_DIR}/endpoint_accuracy.csv" full; then
    echo "J failed preregistered ID-retention gate for seed ${SEED}" >&2
    exit 34
fi

TRANSITION_DIR="${REPLICATION_ROOT}/j_transition_l12to17_n8192"
if [[ ! -f "${TRANSITION_DIR}/summary.json" ]] || ! grep -q '"status": "complete"' "${TRANSITION_DIR}/summary.json"; then
    wait_for_resources
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
        "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
        --checkpoint "${SELECTED_CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants raw full \
        --lengths 12 13 14 15 16 17 \
        --batch-size 128 --batches 64 --seed 584001 --device cuda \
        --out-dir "${TRANSITION_DIR}"
fi

CONTROL_DIR="${REPLICATION_ROOT}/j_controls_l10to17_n1024"
if [[ ! -f "${CONTROL_DIR}/summary.json" ]] || ! grep -q '"status": "complete"' "${CONTROL_DIR}/summary.json"; then
    wait_for_resources
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
        "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
        --checkpoint "${SELECTED_CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants raw full no_AB identity_D full_executor_off \
        --lengths 10 11 12 13 14 15 16 17 \
        --batch-size 128 --batches 8 --seed 584001 --device cuda \
        --out-dir "${CONTROL_DIR}"
fi

WIDE_DIR="${REPLICATION_ROOT}/j_raw_full_l1to30_n512"
if [[ ! -f "${WIDE_DIR}/summary.json" ]] || ! grep -q '"status": "complete"' "${WIDE_DIR}/summary.json"; then
    wait_for_resources
    CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH="${CODE_DIR}" \
        "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
        --checkpoint "${SELECTED_CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants raw full \
        --lengths {1..30} \
        --batch-size 128 --batches 4 --seed 584001 --device cuda \
        --out-dir "${WIDE_DIR}"
fi

MATRIX_DIR="${REPLICATION_ROOT}/matrix_analysis"
if [[ ! -f "${MATRIX_DIR}/controller_analysis.json" ]] || ! grep -q '"status": "complete"' "${MATRIX_DIR}/controller_analysis.json"; then
    "${PYTHON}" "${CODE_DIR}/scripts/analyze_addition_diag_lora_controllers.py" \
        --controller "seed${SEED}=${CONTROLLER}" \
        --out-dir "${MATRIX_DIR}"
fi

echo "$(date -Is) replication_complete seed=${SEED} selected_step=${SELECTED_STEP} controller=${CONTROLLER}"
