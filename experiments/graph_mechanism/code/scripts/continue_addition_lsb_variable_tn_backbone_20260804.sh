#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 RESUME_STEP TARGET_STEP" >&2
    exit 2
fi

RESUME_STEP="$1"
TARGET_STEP="$2"
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804
OUT_DIR="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0"
CHECKPOINT="${OUT_DIR}/checkpoint_$(printf '%06d' "${RESUME_STEP}").pt"
PYTHON=/data/wujiaju/.venvs/loopreasoner/bin/python

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing resume checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi
if (( TARGET_STEP <= RESUME_STEP )); then
    echo "target step must exceed resume step" >&2
    exit 4
fi

exec "${PYTHON}" -u -m reasoning_loop.paper_length_telomere backbone \
    --task addition \
    --supervision adaptive_step \
    --resume "${CHECKPOINT}" \
    --steps "${TARGET_STEP}" \
    --schedule-total-steps 100001 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --official-model-config \
    --attention-mode causal \
    --position-embedding none \
    --position-injection initial_only \
    --addition-lsb-first \
    --addition-answer-supervision logical_digits \
    --train-max-length 10 \
    --curriculum-interval 1600 \
    --task-step-offset 0 \
    --batch-shared-logical-length \
    --seed 0 \
    --device cuda \
    --amp \
    --log-every 100 \
    --eval-every 1000 \
    --eval-batch-size 256 \
    --eval-batches 4 \
    --checkpoint-every 1000 \
    --out-dir "${OUT_DIR}"
