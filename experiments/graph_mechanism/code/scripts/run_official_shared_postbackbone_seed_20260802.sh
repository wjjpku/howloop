#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 TASK BACKBONE_SEED PHYSICAL_GPU" >&2
    exit 2
fi

TASK="$1"
BACKBONE_SEED="$2"
PHYSICAL_GPU="$3"
if ! [[ "${BACKBONE_SEED}" =~ ^[0-9]+$ ]]; then
    echo "BACKBONE_SEED must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
    echo "PHYSICAL_GPU must be in [0,7]" >&2
    exit 2
fi

case "${TASK}" in
    copy)
        SLOT=0
        NATIVE_LOGICAL_MAXIMUM=19
        DECLARED_PEAK_MIB=8192
        MEMORY_FRACTION=0.085
        DIAGNOSIS_LENGTHS="19 25 30 40 50 75 100"
        DIAGNOSIS_MAXIMUM_STEP=132
        DIAGNOSIS_EXTENSION_GATE_LENGTH=50
        AUDIT_LENGTHS="19 25 30 40 50 60 75 100"
        AUDIT_MAXIMUM_STEP=132
        ;;
    addition)
        SLOT=1
        NATIVE_LOGICAL_MAXIMUM=19
        DECLARED_PEAK_MIB=12288
        MEMORY_FRACTION=0.130
        DIAGNOSIS_LENGTHS="19 25 30 40 50 60 75 100"
        DIAGNOSIS_MAXIMUM_STEP=133
        DIAGNOSIS_EXTENSION_GATE_LENGTH=30
        AUDIT_LENGTHS="19 25 30 40 50 60 75 100"
        AUDIT_MAXIMUM_STEP=133
        ;;
    sum_reverse)
        SLOT=2
        NATIVE_LOGICAL_MAXIMUM=19
        DECLARED_PEAK_MIB=4096
        MEMORY_FRACTION=0.040
        DIAGNOSIS_LENGTHS="19 24 30 40 50 75 100"
        DIAGNOSIS_MAXIMUM_STEP=132
        DIAGNOSIS_EXTENSION_GATE_LENGTH=100
        AUDIT_LENGTHS="19 24 30 40 50 60 75 100"
        AUDIT_MAXIMUM_STEP=132
        ;;
    *)
        echo "TASK must be copy, addition, or sum_reverse" >&2
        exit 2
        ;;
esac

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
RUNNER=/data/wujiaju/LooPlus/scripts/run_paper_length_telomere_remote_20260731.sh
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
LOCK_ROOT="${RUN_ROOT}/locks"
LABEL="${TASK}_adaptive_step_official_seed${BACKBONE_SEED}"
BACKBONE_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/official_formal.json"
DIAGNOSIS_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/diagnose.json"
DIAGNOSIS_SUMMARY="${RUN_ROOT}/diagnosis/${LABEL}/summary.json"
CONTROLLER_SEED=211001
AUDIT_SEED=261001
RESERVE_MIB=16384
mkdir -p "${LOCK_ROOT}" "${LOG_ROOT}"

manifest_is_complete() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "complete"' "${manifest}"
}

manifest_is_failed() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "failed"' "${manifest}"
}

wait_for_backbone() {
    while ! manifest_is_complete "${BACKBONE_MANIFEST}"; do
        if manifest_is_failed "${BACKBONE_MANIFEST}"; then
            echo "$(date -Is) backbone failed task=${TASK} seed=${BACKBONE_SEED}" >&2
            return 1
        fi
        echo "$(date -Is) waiting for backbone task=${TASK} seed=${BACKBONE_SEED}"
        sleep 60
    done
}

gpu_snapshot() {
    local free_mib
    local pid_memory
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    pid_memory="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed 's/ MiB//g' | sort -n | tr '\n' ';' | sed 's/;$//')"
    printf '%s|%s\n' "${free_mib}" "${pid_memory}"
}

wait_for_stable_shared_gate() {
    local required=$((RESERVE_MIB + DECLARED_PEAK_MIB))
    local first second first_free second_free
    while true; do
        first="$(gpu_snapshot)"
        sleep 61
        second="$(gpu_snapshot)"
        first_free="${first%%|*}"
        second_free="${second%%|*}"
        if [[ "${first#*|}" == "${second#*|}" ]] \
            && [[ "${first_free}" -ge "${required}" ]] \
            && [[ "${second_free}" -ge "${required}" ]]; then
            echo "$(date -Is) shared gate passed task=${TASK} seed=${BACKBONE_SEED} gpu=${PHYSICAL_GPU} first=${first} second=${second}"
            return 0
        fi
        echo "$(date -Is) shared gate waiting task=${TASK} seed=${BACKBONE_SEED} gpu=${PHYSICAL_GPU} first=${first} second=${second} required=${required}"
    done
}

run_shared_stage() {
    local action="$1"
    local manifest="$2"
    shift 2
    local attempt=0
    local status=0
    if manifest_is_complete "${manifest}"; then
        echo "$(date -Is) stage already complete action=${action} task=${TASK} seed=${BACKBONE_SEED}"
        return 0
    fi
    while true; do
        attempt=$((attempt + 1))
        set +e
        (
            flock -x 8
            flock -x 9
            wait_for_stable_shared_gate
            env \
                BASELINE_VARIANT=official \
                CONTROLLER_SEED="${CONTROLLER_SEED}" \
                AUDIT_SEED="${AUDIT_SEED}" \
                ALLOW_SHARED_GPU=1 \
                DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB}" \
                RESERVE_MIB="${RESERVE_MIB}" \
                PAPER_CUDA_MEMORY_FRACTION="${MEMORY_FRACTION}" \
                "$@" \
                bash "${RUNNER}" "${action}" "${TASK}" adaptive_step \
                    "${PHYSICAL_GPU}" "${BACKBONE_SEED}"
        ) 8>"${LOCK_ROOT}/paper_job_slot${SLOT}.lock" 9>"${LOCK_ROOT}/gpu${PHYSICAL_GPU}.lock"
        status=$?
        set -e
        if [[ "${status}" -eq 0 ]]; then
            return 0
        fi
        if [[ "${status}" -ne 75 && "${status}" -ne 143 ]]; then
            echo "$(date -Is) deterministic stage failure action=${action} task=${TASK} seed=${BACKBONE_SEED} exit=${status}" >&2
            return "${status}"
        fi
        if [[ "${attempt}" -ge 5 ]]; then
            echo "$(date -Is) transient stage retries exhausted action=${action} task=${TASK} seed=${BACKBONE_SEED}" >&2
            return "${status}"
        fi
        sleep 30
    done
}

wait_for_backbone

run_shared_stage diagnose "${DIAGNOSIS_MANIFEST}" \
    DIAGNOSIS_BATCH_SIZE=64 \
    DIAGNOSIS_BATCHES=8 \
    DIAGNOSIS_LENGTHS="${DIAGNOSIS_LENGTHS}" \
    DIAGNOSIS_MAXIMUM_STEP="${DIAGNOSIS_MAXIMUM_STEP}" \
    DIAGNOSIS_EXTENSION_GATE_LENGTH="${DIAGNOSIS_EXTENSION_GATE_LENGTH}"

GATE_PASSED="$("${PYTHON_BIN}" - "${DIAGNOSIS_SUMMARY}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if payload["disease_gate"]["passed"] else "0")
PY
)"
if [[ "${GATE_PASSED}" != 1 ]]; then
    echo "$(date -Is) telomere gate negative; J intentionally skipped task=${TASK} seed=${BACKBONE_SEED} diagnosis=${DIAGNOSIS_SUMMARY}"
    exit 0
fi

for LOGICAL_MAXIMUM in "${NATIVE_LOGICAL_MAXIMUM}" 40; do
    CONDITION_LABEL="${LABEL}_rank48_logical1to${LOGICAL_MAXIMUM}_seed${CONTROLLER_SEED}"
    CONTROLLER_MANIFEST="${RUN_ROOT}/manifests/${CONDITION_LABEL}/controller.json"
    AUDIT_MANIFEST="${RUN_ROOT}/manifests/${CONDITION_LABEL}/audit.json"
    run_shared_stage controller "${CONTROLLER_MANIFEST}" \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${LOGICAL_MAXIMUM}"
    run_shared_stage audit "${AUDIT_MANIFEST}" \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${LOGICAL_MAXIMUM}" \
        AUDIT_BATCH_SIZE=32 \
        AUDIT_BATCHES=16 \
        AUDIT_LENGTHS="${AUDIT_LENGTHS}" \
        AUDIT_MAXIMUM_STEP="${AUDIT_MAXIMUM_STEP}" \
        AUDIT_MODES="full no_AB identity_D"
done

echo "$(date -Is) post-backbone experiment complete task=${TASK} seed=${BACKBONE_SEED}"
