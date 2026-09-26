#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[0-2]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU BACKBONE_SEED" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
BACKBONE_SEED="$2"
CONTROLLER_SEED=$((211001 + BACKBONE_SEED))
AUDIT_SEED=$((261001 + BACKBONE_SEED))
CODE_DIR=/data/paperexperiment/LooPlus
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
RUNNER="${CODE_DIR}/scripts/run_paper_length_telomere_remote_20260731.sh"
BASELINE_LABEL="copy4_adaptive_step_official_seed${BACKBONE_SEED}"
CONTROLLER_LABEL="${BASELINE_LABEL}_rank48_identitywarmup_logical20to40_anchor1_postfinal_wsd_lr3em4_warmup2048_stable2816_5k_cseed${CONTROLLER_SEED}"
PIPELINE_LOG="${LOG_ROOT}/${CONTROLLER_LABEL}_pipeline.log"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/manifests/${CONTROLLER_LABEL}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export BASELINE_VARIANT=official
export ALLOW_SHARED_GPU=1
export DECLARED_PEAK_MIB=6144
export RESERVE_MIB=16384

echo "$(date -Is) copy4 pipeline start seed=${BACKBONE_SEED} gpu=${PHYSICAL_GPU}"

if [[ ! -f "${RUN_ROOT}/backbones/${BASELINE_LABEL}/summary.json" ]] || \
   ! grep -q '"status": "complete"' "${RUN_ROOT}/backbones/${BASELINE_LABEL}/summary.json"; then
    bash "${RUNNER}" official_formal copy4 adaptive_step \
        "${PHYSICAL_GPU}" "${BACKBONE_SEED}"
else
    echo "$(date -Is) baseline already complete seed=${BACKBONE_SEED}"
fi

DIAGNOSIS_BATCH_SIZE=128 \
DIAGNOSIS_BATCHES=4 \
DIAGNOSIS_LENGTHS="10 19 20 25 30 35 40 50 60" \
DIAGNOSIS_MAXIMUM_STEP=92 \
DIAGNOSIS_EXTENSION_GATE_LENGTH=40 \
bash "${RUNNER}" diagnose copy4 adaptive_step \
    "${PHYSICAL_GPU}" "${BACKBONE_SEED}"

CONTROLLER_CURRICULUM=logical_range \
CONTROLLER_LOGICAL_MAX_LENGTH=40 \
CONTROLLER_LABEL_OVERRIDE="${CONTROLLER_LABEL}" \
CONTROLLER_SEED="${CONTROLLER_SEED}" \
CONTROLLER_TRAINING_PROFILE=identity_long_warmup \
CONTROLLER_INITIALIZATION=identity \
CONTROLLER_FORCE=1 \
CONTROLLER_ANCHOR_STEP=1 \
CONTROLLER_WARMUP_UPDATES=2048 \
CONTROLLER_FINAL_LR_RATIO=0.1 \
CONTROLLER_LR_SCHEDULE=wsd \
CONTROLLER_STABLE_UPDATES=2816 \
CONTROLLER_POST_FINAL_J=1 \
CONTROLLER_DENSE_STAGE_COUNT=0 \
CONTROLLER_GRAD_CLIP=1.0 \
CONTROLLER_LEARNING_RATE_MULTIPLIER=15.0 \
CONTROLLER_DIAGONAL_LR_MULTIPLIER=0.1 \
CONTROLLER_STAGE_ROUND_MULTIPLIER=3 \
bash "${RUNNER}" controller copy4 adaptive_step \
    "${PHYSICAL_GPU}" "${BACKBONE_SEED}"

CONTROLLER_CURRICULUM=logical_range \
CONTROLLER_LOGICAL_MAX_LENGTH=40 \
CONTROLLER_LABEL_OVERRIDE="${CONTROLLER_LABEL}" \
CONTROLLER_SEED="${CONTROLLER_SEED}" \
AUDIT_SEED="${AUDIT_SEED}" \
AUDIT_BATCH_SIZE=128 \
AUDIT_BATCHES=4 \
AUDIT_LENGTHS="1 10 19 20 25 30 35 40 50 60" \
AUDIT_MAXIMUM_STEP=92 \
AUDIT_MODES="full no_AB identity_D no_bias" \
AUDIT_POST_FINAL_J=1 \
bash "${RUNNER}" audit copy4 adaptive_step \
    "${PHYSICAL_GPU}" "${BACKBONE_SEED}"

echo "$(date -Is) copy4 pipeline complete seed=${BACKBONE_SEED} gpu=${PHYSICAL_GPU}"
