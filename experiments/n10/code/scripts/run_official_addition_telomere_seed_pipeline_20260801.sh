#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 BACKBONE_SEED" >&2
    exit 2
fi

BACKBONE_SEED="$1"
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
WAITER=/data/paperexperiment/LooPlus/scripts/wait_for_empty_gpu_paper_length_telomere_20260731.sh
LABEL="addition_adaptive_step_official_seed${BACKBONE_SEED}"
BACKBONE_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/official_formal.json"
DIAGNOSIS_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/diagnose.json"
CONTROLLER_SEED=211001

export BASELINE_VARIANT=official
export CONTROLLER_SEED
export AUDIT_SEED=261001
export DIAGNOSIS_BATCH_SIZE=64
export DIAGNOSIS_BATCHES=8
export DIAGNOSIS_LENGTHS="19 30 40 50 75 100"
export DIAGNOSIS_MAXIMUM_STEP=133
export DIAGNOSIS_EXTENSION_GATE_LENGTH=30

manifest_is_complete() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "complete"' "${manifest}"
}

run_stage() {
    local action="$1"
    local manifest="$2"
    local maximum_attempts="${3:-5}"
    local attempt=0
    local status=0
    while ! manifest_is_complete "${manifest}"; do
        attempt=$((attempt + 1))
        set +e
        bash "${WAITER}" "${action}" addition adaptive_step "${BACKBONE_SEED}"
        status=$?
        set -e
        if [[ "${status}" -ne 0 ]]; then
            echo "$(date -Is) stage=${action} task=addition seed=${BACKBONE_SEED} attempt=${attempt} exit=${status}"
            if [[ "${status}" -ne 75 && "${status}" -ne 143 ]]; then
                echo "stage ${action} failed deterministically; not retrying" >&2
                return "${status}"
            fi
            if [[ "${attempt}" -ge "${maximum_attempts}" ]]; then
                echo "stage ${action} exhausted ${maximum_attempts} attempts" >&2
                return "${status}"
            fi
            sleep 30
        fi
    done
}

run_stage official_formal "${BACKBONE_MANIFEST}" 5
run_stage diagnose "${DIAGNOSIS_MANIFEST}" 3

run_logical_horizon_condition() {
    local logical_maximum="$1"
    local condition_label="${LABEL}_rank48_logical1to${logical_maximum}_seed${CONTROLLER_SEED}"
    local controller_manifest="${RUN_ROOT}/manifests/${condition_label}/controller.json"
    local audit_manifest="${RUN_ROOT}/manifests/${condition_label}/audit.json"

    export CONTROLLER_CURRICULUM=logical_range
    export CONTROLLER_LOGICAL_MAX_LENGTH="${logical_maximum}"
    export AUDIT_BATCH_SIZE=32
    export AUDIT_BATCHES=16
    export AUDIT_LENGTHS="19 25 30 40 50 60 75 100"
    export AUDIT_MAXIMUM_STEP=133
    export AUDIT_MODES="full no_AB identity_D"

    run_stage controller "${controller_manifest}" 3
    run_stage audit "${audit_manifest}" 3
}

run_logical_horizon_condition 19
run_logical_horizon_condition 40

echo "$(date -Is) official addition telomere seed pipeline complete seed=${BACKBONE_SEED}"
