#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 BACKBONE_CHECKPOINT CONTROLLER_OUT_DIR" >&2
    exit 2
fi

CHECKPOINT="$1"
OUT_DIR="$2"
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing backbone checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi
if [[ -e "${OUT_DIR}" ]]; then
    echo "refusing to overwrite controller directory: ${OUT_DIR}" >&2
    exit 4
fi

exec "${PYTHON}" -u -m reasoning_loop.paper_length_telomere controller \
    --checkpoint "${CHECKPOINT}" \
    --controller-parameterization diagonal_low_rank \
    --rank 48 \
    --seed 521101 \
    --device cuda \
    --grad-clip 1.0 \
    --learning-rate-multiplier 5.0 \
    --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-curriculum logical_range \
    --controller-logical-min-length 2 \
    --controller-logical-max-length 10 \
    --controller-anchor-step 1 \
    --controller-warmup-updates 2048 \
    --controller-stable-updates 2816 \
    --controller-final-lr-ratio 0.1 \
    --controller-lr-schedule wsd \
    --controller-ce-temperature 1.0 \
    --controller-supervision full_answer \
    --stage-round-multiplier 3 \
    --controller-checkpoint-every 256 \
    --no-controller-post-final-j \
    --force \
    --out-dir "${OUT_DIR}"
