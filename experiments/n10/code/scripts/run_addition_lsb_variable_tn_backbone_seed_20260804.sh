#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 SEED" >&2
    exit 2
fi

SEED="$1"
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "seed must be a non-negative integer" >&2
    exit 2
fi

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804
OUT_DIR="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed${SEED}"
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
CODE_DIR=/data/paperexperiment/LooPlus

if [[ -e "${OUT_DIR}" ]]; then
    echo "refusing to overwrite backbone directory: ${OUT_DIR}" >&2
    exit 3
fi

cd "${CODE_DIR}"
exec "${PYTHON}" -u -m reasoning_loop.paper_length_telomere backbone \
    --task addition \
    --supervision adaptive_step \
    --steps 80000 \
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
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --log-every 100 \
    --eval-every 1000 \
    --eval-batch-size 256 \
    --eval-batches 4 \
    --checkpoint-every 5000 \
    --out-dir "${OUT_DIR}"
