#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
RUNNER=/data/paperexperiment/LooPlus/scripts/run_paper_length_telomere_remote_20260731.sh
PIPELINE_LOG=/data/paperexperiment/logs/paper_length_telomere_20260731/official_addition_j_horizon_shared_20260804.log
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384

mkdir -p "$(dirname "${PIPELINE_LOG}")"

manifest_complete() {
    local path="$1"
    [[ -f "${path}" ]] && grep -q '"status": "complete"' "${path}"
}

run_condition() {
    local seed="$1" logical_maximum="$2"
    local label controller_manifest audit_manifest
    label="addition_adaptive_step_official_seed${seed}_rank48_logical1to${logical_maximum}_seed211001"
    controller_manifest="${RUN_ROOT}/manifests/${label}/controller.json"
    audit_manifest="${RUN_ROOT}/manifests/${label}/audit.json"

    if ! manifest_complete "${controller_manifest}"; then
        echo "$(date -Is) controller_start seed=${seed} logical_max=${logical_maximum}" | tee -a "${PIPELINE_LOG}"
        BASELINE_VARIANT=official \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${logical_maximum}" \
        CONTROLLER_SEED=211001 \
        CONTROLLER_FORCE=1 \
        CONTROLLER_TRAINING_PROFILE=legacy \
        CONTROLLER_INITIALIZATION=dense_svd \
        CONTROLLER_DENSE_STAGE_COUNT=2 \
        CONTROLLER_POST_FINAL_J=0 \
        ALLOW_SHARED_GPU=1 \
        DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB}" \
        RESERVE_MIB="${RESERVE_MIB}" \
            bash "${RUNNER}" controller addition adaptive_step "${PHYSICAL_GPU}" "${seed}"
    fi

    if ! manifest_complete "${audit_manifest}"; then
        echo "$(date -Is) audit_start seed=${seed} logical_max=${logical_maximum}" | tee -a "${PIPELINE_LOG}"
        BASELINE_VARIANT=official \
        CONTROLLER_CURRICULUM=logical_range \
        CONTROLLER_LOGICAL_MAX_LENGTH="${logical_maximum}" \
        CONTROLLER_SEED=211001 \
        AUDIT_SEED=261001 \
        AUDIT_BATCH_SIZE=32 \
        AUDIT_BATCHES=16 \
        AUDIT_LENGTHS="19 25 30 40 50 60 75 100" \
        AUDIT_MAXIMUM_STEP=133 \
        AUDIT_MODES="full no_AB identity_D" \
        AUDIT_POST_FINAL_J=0 \
        ALLOW_SHARED_GPU=1 \
        DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB}" \
        RESERVE_MIB="${RESERVE_MIB}" \
            bash "${RUNNER}" audit addition adaptive_step "${PHYSICAL_GPU}" "${seed}"
    fi
    echo "$(date -Is) condition_complete seed=${seed} logical_max=${logical_maximum}" | tee -a "${PIPELINE_LOG}"
}

for seed in 0 1 2; do
    run_condition "${seed}" 19
    run_condition "${seed}" 40
done

echo "$(date -Is) all_conditions_complete" | tee -a "${PIPELINE_LOG}"
