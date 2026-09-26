#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 {lsb_nope|lsb_posabs|msb_full_posabs}" >&2
    exit 2
fi

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/reverse_addition_20260804
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
VARIANT="$1"

case "${VARIANT}" in
    lsb_nope)
        BACKBONE_LABEL=addition_lsb_fixed_n10_t11_nope_seed0
        ;;
    lsb_posabs)
        BACKBONE_LABEL=addition_lsb_fixed_n10_t11_posabs_seed0
        ;;
    msb_full_posabs)
        BACKBONE_LABEL=addition_msb_fixed_n10_t11_fullattn_posabs_seed0
        ;;
    *)
        echo "unknown variant: ${VARIANT}" >&2
        exit 2
        ;;
esac

CHECKPOINT="${RUN_ROOT}/backbones/${BACKBONE_LABEL}/selected.pt"
SELECTION="${RUN_ROOT}/backbones/${BACKBONE_LABEL}/selection.json"
CONTROLLER_DIR="${RUN_ROOT}/controllers/${BACKBONE_LABEL}_rank48_identity_fullanswer_l1to10_anchor1_wsd5376_seed521001"
AUDIT_DIR="${RUN_ROOT}/audits/${BACKBONE_LABEL}_rank48_identity_fullanswer_l1to10_anchor1_wsd5376_seed521001_l1to20"

if [[ ! -f "${CHECKPOINT}" ]] || ! grep -q '"status": "selected"' "${SELECTION}"; then
    echo "backbone has no selected just-converged checkpoint: ${BACKBONE_LABEL}" >&2
    exit 3
fi
if [[ -e "${CONTROLLER_DIR}" ]] || [[ -e "${AUDIT_DIR}" ]]; then
    echo "refusing to overwrite an existing controller or audit directory" >&2
    exit 4
fi

"${PYTHON}" -u -m reasoning_loop.paper_length_telomere controller \
    --checkpoint "${CHECKPOINT}" \
    --controller-parameterization diagonal_low_rank \
    --rank 48 \
    --seed 521001 \
    --device auto \
    --grad-clip 1.0 \
    --learning-rate-multiplier 5.0 \
    --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-curriculum logical_range \
    --controller-logical-min-length 1 \
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
    --out-dir "${CONTROLLER_DIR}"

"${PYTHON}" -u -m reasoning_loop.paper_length_telomere audit \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER_DIR}/controller.pt" \
    --lengths {1..20} \
    --batch-size 128 \
    --batches 4 \
    --maximum-step 25 \
    --modes full no_AB identity_D \
    --seed 584001 \
    --device auto \
    --no-post-final-j \
    --out-dir "${AUDIT_DIR}"
