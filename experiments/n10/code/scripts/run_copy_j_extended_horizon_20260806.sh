#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^(60|80)$ ]] || ! [[ "$2" =~ ^[0-7]$ ]]; then
    echo "usage: $0 {60|80} PHYSICAL_GPU" >&2
    exit 2
fi

MAX_LENGTH="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/paperexperiment/LooPlus
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
RUNNER="${CODE_DIR}/scripts/run_paper_length_telomere_remote_20260731.sh"
CONTROLLER_SEED=211001
AUDIT_SEED=261001
CONTROLLER_LABEL="copy_adaptive_step_official_seed0_rank48_identitywarmup_logical20to${MAX_LENGTH}_explicitmin20_anchor1_postfinal_wsd_lr3em4_warmup2048_stable2816_5k_seed${CONTROLLER_SEED}"
PIPELINE_LOG="${LOG_ROOT}/${CONTROLLER_LABEL}_pipeline.log"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/manifests/${CONTROLLER_LABEL}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

common_environment=(
    BASELINE_VARIANT=official
    ALLOW_SHARED_GPU=1
    DECLARED_PEAK_MIB=8192
    RESERVE_MIB=16384
    CONTROLLER_CURRICULUM=logical_range
    CONTROLLER_LOGICAL_MIN_LENGTH=20
    CONTROLLER_LOGICAL_MAX_LENGTH="${MAX_LENGTH}"
    CONTROLLER_LABEL_OVERRIDE="${CONTROLLER_LABEL}"
    CONTROLLER_SEED="${CONTROLLER_SEED}"
    CONTROLLER_TRAINING_PROFILE=identity_long_warmup
    CONTROLLER_INITIALIZATION=identity
    CONTROLLER_FORCE=1
    CONTROLLER_ANCHOR_STEP=1
    CONTROLLER_WARMUP_UPDATES=2048
    CONTROLLER_STABLE_UPDATES=2816
    CONTROLLER_FINAL_LR_RATIO=0.1
    CONTROLLER_LR_SCHEDULE=wsd
    CONTROLLER_POST_FINAL_J=1
    CONTROLLER_DENSE_STAGE_COUNT=0
    CONTROLLER_GRAD_CLIP=1.0
    CONTROLLER_LEARNING_RATE_MULTIPLIER=15.0
    CONTROLLER_DIAGONAL_LR_MULTIPLIER=0.1
    CONTROLLER_STAGE_ROUND_MULTIPLIER=3
)

echo "$(date -Is) Copy extended-horizon J pipeline start max_length=${MAX_LENGTH} gpu=${PHYSICAL_GPU}"

env "${common_environment[@]}" \
    bash "${RUNNER}" controller copy adaptive_step "${PHYSICAL_GPU}" 0

env "${common_environment[@]}" \
    AUDIT_SEED="${AUDIT_SEED}" \
    AUDIT_BATCH_SIZE=128 \
    AUDIT_BATCHES=4 \
    AUDIT_LENGTHS="19 20 25 30 40 50 55 60 65 70 75 80 90 100 120" \
    AUDIT_MAXIMUM_STEP=132 \
    AUDIT_MODES="full no_AB identity_D no_bias" \
    AUDIT_POST_FINAL_J=1 \
    bash "${RUNNER}" audit copy adaptive_step "${PHYSICAL_GPU}" 0

echo "$(date -Is) Copy extended-horizon J pipeline complete max_length=${MAX_LENGTH} gpu=${PHYSICAL_GPU}"
