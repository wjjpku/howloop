#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]] \
    || ! [[ "$1" =~ ^[0-9]+$ ]] \
    || ! [[ "$2" =~ ^[0-7]$ ]] \
    || [[ "$#" -eq 3 && "$3" != "--launch-backbone" ]]; then
    echo "usage: $0 BACKBONE_SEED PHYSICAL_GPU [--launch-backbone]" >&2
    exit 2
fi

BACKBONE_SEED="$1"
PHYSICAL_GPU="$2"
LAUNCH_BACKBONE=false
if [[ "$#" -eq 3 ]]; then
    LAUNCH_BACKBONE=true
fi
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
RUNNER=/data/paperexperiment/LooPlus/scripts/run_paper_length_telomere_remote_20260731.sh
LABEL="parity_adaptive_step_released64_seed${BACKBONE_SEED}"
BACKBONE_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/released_formal.json"
CONTROLLER_SEED=211001
AUDIT_SEED=261001
DECLARED_PEAK_MIB=1536
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

manifest_is_complete() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "complete"' "${manifest}"
}

manifest_is_failed() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "failed"' "${manifest}"
}

manifest_is_launching() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "launching"' "${manifest}"
}

wait_for_backbone() {
    while ! manifest_is_complete "${BACKBONE_MANIFEST}"; do
        if manifest_is_failed "${BACKBONE_MANIFEST}"; then
            echo "$(date -Is) backbone manifest failed: ${BACKBONE_MANIFEST}" >&2
            return 1
        fi
        echo "$(date -Is) waiting for released64 backbone seed=${BACKBONE_SEED}"
        sleep 60
    done
}

gpu_snapshot() {
    local free_mib
    local pid_memory
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    pid_memory="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed 's/ MiB//g' | tr '\n' ';' | sed 's/;$//')"
    printf '%s|%s\n' "${free_mib}" "${pid_memory}"
}

wait_for_stable_shared_gate() {
    local first
    local second
    local first_free
    local second_free
    while true; do
        first="$(gpu_snapshot)"
        sleep 61
        second="$(gpu_snapshot)"
        first_free="${first%%|*}"
        second_free="${second%%|*}"
        if [[ "${first#*|}" == "${second#*|}" ]] \
            && [[ "${first_free}" -ge "${REQUIRED_FREE_MIB}" ]] \
            && [[ "${second_free}" -ge "${REQUIRED_FREE_MIB}" ]]; then
            echo "$(date -Is) shared gate passed gpu=${PHYSICAL_GPU} first=${first} second=${second}"
            return 0
        fi
        echo "$(date -Is) shared gate retry gpu=${PHYSICAL_GPU} first=${first} second=${second} required_free=${REQUIRED_FREE_MIB}"
    done
}

run_stage() {
    local action="$1"
    local manifest="$2"
    shift 2
    local attempt=0
    local status=0
    while ! manifest_is_complete "${manifest}"; do
        attempt=$((attempt + 1))
        wait_for_stable_shared_gate
        set +e
        env \
            BASELINE_VARIANT=released64 \
            CONTROLLER_SEED="${CONTROLLER_SEED}" \
            AUDIT_SEED="${AUDIT_SEED}" \
            ALLOW_SHARED_GPU=1 \
            DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB}" \
            RESERVE_MIB="${RESERVE_MIB}" \
            PAPER_CUDA_MEMORY_FRACTION=0.01875 \
            "$@" \
            bash "${RUNNER}" "${action}" parity adaptive_step "${PHYSICAL_GPU}" "${BACKBONE_SEED}"
        status=$?
        set -e
        if [[ "${status}" -ne 0 ]]; then
            echo "$(date -Is) stage=${action} seed=${BACKBONE_SEED} attempt=${attempt} exit=${status}" >&2
            if [[ "${status}" -ne 75 && "${status}" -ne 143 ]]; then
                echo "stage ${action} failed deterministically; not retrying" >&2
                return "${status}"
            fi
            if [[ "${attempt}" -ge 5 ]]; then
                return "${status}"
            fi
            sleep 30
        fi
    done
}

run_logical_horizon_condition() {
    local logical_maximum="$1"
    local condition_label="${LABEL}_rank48_logical1to${logical_maximum}_seed${CONTROLLER_SEED}"
    local controller_manifest="${RUN_ROOT}/manifests/${condition_label}/controller.json"
    local audit_manifest="${RUN_ROOT}/manifests/${condition_label}/audit.json"

    run_stage controller "${controller_manifest}" \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${logical_maximum}"
    run_stage audit "${audit_manifest}" \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${logical_maximum}" \
        AUDIT_BATCH_SIZE=64 \
        AUDIT_BATCHES=8 \
        AUDIT_LENGTHS="20 30 40 50 60 75 84 100 120 150 200" \
        AUDIT_MAXIMUM_STEP=200 \
        AUDIT_MODES="full no_AB identity_D"
}

if [[ "${LAUNCH_BACKBONE}" == true ]] && ! manifest_is_launching "${BACKBONE_MANIFEST}"; then
    run_stage released_formal "${BACKBONE_MANIFEST}"
else
    wait_for_backbone
fi

run_stage diagnose "${RUN_ROOT}/manifests/${LABEL}/diagnose.json" \
    DIAGNOSIS_LENGTHS="20 40 50 75 100" \
    DIAGNOSIS_EXTENSION_GATE_LENGTH=100

run_logical_horizon_condition 20
run_logical_horizon_condition 40

echo "$(date -Is) released64 shared primary postbackbone complete seed=${BACKBONE_SEED} gpu=${PHYSICAL_GPU}"
