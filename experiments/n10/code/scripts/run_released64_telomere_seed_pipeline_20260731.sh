#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 BACKBONE_SEED" >&2
    exit 2
fi

BACKBONE_SEED="$1"
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
WAITER=/data/paperexperiment/LooPlus/scripts/wait_for_empty_gpu_paper_length_telomere_20260731.sh
LABEL="parity_adaptive_step_released64_seed${BACKBONE_SEED}"
BACKBONE_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/released_formal.json"
DIAGNOSIS_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/diagnose.json"
CONTROLLER_SEED=211001
CONTROLLER_CURRICULUM=extension
CONTROLLER_LABEL="${LABEL}_rank48_${CONTROLLER_CURRICULUM}_seed${CONTROLLER_SEED}"
CONTROLLER_MANIFEST="${RUN_ROOT}/manifests/${CONTROLLER_LABEL}/controller.json"
AUDIT_MANIFEST="${RUN_ROOT}/manifests/${CONTROLLER_LABEL}/audit.json"

export BASELINE_VARIANT=released64
export CONTROLLER_CURRICULUM
export CONTROLLER_SEED
export AUDIT_SEED=261001
export DIAGNOSIS_LENGTHS="20 40 50 75 100"
export DIAGNOSIS_EXTENSION_GATE_LENGTH=100

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
        bash "${WAITER}" "${action}" parity adaptive_step "${BACKBONE_SEED}"
        status=$?
        set -e
        if [[ "${status}" -ne 0 ]]; then
            echo "$(date -Is) stage=${action} seed=${BACKBONE_SEED} attempt=${attempt} exit=${status}"
            if [[ "${attempt}" -ge "${maximum_attempts}" ]]; then
                echo "stage ${action} exhausted ${maximum_attempts} attempts" >&2
                return "${status}"
            fi
            sleep 30
        fi
    done
}

run_stage released_formal "${BACKBONE_MANIFEST}" 5
run_stage diagnose "${DIAGNOSIS_MANIFEST}" 3

run_logical_horizon_condition() {
    local logical_max="$1"
    local logical_label="${LABEL}_rank48_logical1to${logical_max}_seed${CONTROLLER_SEED}"
    local controller_manifest="${RUN_ROOT}/manifests/${logical_label}/controller.json"
    local audit_manifest="${RUN_ROOT}/manifests/${logical_label}/audit.json"

    export CONTROLLER_CURRICULUM=logical_range
    export CONTROLLER_LOGICAL_MAX_LENGTH="${logical_max}"
    # Horizon-generalization screening keeps the optimizer-update budget
    # matched between J_1-20 and J_1-40, then evaluates a denser length grid.
    export AUDIT_BATCH_SIZE=64
    export AUDIT_BATCHES=8
    export AUDIT_LENGTHS="20 30 40 50 60 75 84 100 120 150 200"
    export AUDIT_MAXIMUM_STEP=200
    export AUDIT_MODES="full no_AB identity_D"

    run_stage controller "${controller_manifest}" 3
    run_stage audit "${audit_manifest}" 3
}

run_logical_horizon_condition 20
run_logical_horizon_condition 40

# The logical-horizon comparison is the primary telomere experiment.  Run the
# older post-anchor extension controller only after J_1-20/J_1-40 so that the
# next scarce empty-GPU window answers the registered primary question first.
export CONTROLLER_CURRICULUM=extension
unset CONTROLLER_LOGICAL_MAX_LENGTH || true
export AUDIT_BATCH_SIZE=128
export AUDIT_BATCHES=32
export AUDIT_LENGTHS="20 40 50 75 84 100"
export AUDIT_MAXIMUM_STEP=132
export AUDIT_MODES="full no_AB identity_D mean_D no_bias shuffle_D"
run_stage controller "${CONTROLLER_MANIFEST}" 3
run_stage audit "${AUDIT_MANIFEST}" 3

echo "$(date -Is) released64 telomere seed pipeline complete seed=${BACKBONE_SEED}"
