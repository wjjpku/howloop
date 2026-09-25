#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 TRAINING_PID" >&2
    exit 2
fi

TRAINING_PID="$1"
PYTHON=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804
CHECKPOINT="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
CONTROLLER="${RUN_ROOT}/controllers/addition_lsb_variable_m1to10_tn80k_rank48_identity_logicaldigits_m2to10_anchor1_wsd14336_stable10000_seed521101/controller.pt"
AUDIT_ROOT="${RUN_ROOT}/audits"

while kill -0 "${TRAINING_PID}" 2>/dev/null; do
    sleep 30
done

if [[ ! -f "${CONTROLLER}" ]]; then
    echo "controller training exited without final artifact" >&2
    exit 31
fi

cd /data/wujiaju/LooPlus

ID_DIR="${AUDIT_ROOT}/final80k_j14336_id_l1to10_n1024"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER}" \
    --variants raw full \
    --lengths 1 2 3 4 5 6 7 8 9 10 \
    --batch-size 128 \
    --batches 8 \
    --seed 584001 \
    --device cuda \
    --out-dir "${ID_DIR}"

awk -F, '
    NR == 1 {
        for (i = 1; i <= NF; i++) {
            if ($i == "supervised_digit_exact_match") metric = i
        }
        next
    }
    $1 == "full" && $metric < 0.995 { failed = 1 }
    END { exit failed }
' "${ID_DIR}/endpoint_accuracy.csv" || {
    echo "J failed the preregistered ID gate; OOD remains unopened" >&2
    exit 32
}

TRANSITION_DIR="${AUDIT_ROOT}/final80k_j14336_transition_l12to17_n8192"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER}" \
    --variants raw full \
    --lengths 12 13 14 15 16 17 \
    --batch-size 128 \
    --batches 64 \
    --seed 584001 \
    --device cuda \
    --out-dir "${TRANSITION_DIR}"

WIDE_DIR="${AUDIT_ROOT}/final80k_j14336_raw_full_l1to30"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m scripts.evaluate_addition_controller_endpoint_accuracy \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER}" \
    --variants raw full \
    --lengths \
        1 2 3 4 5 6 7 8 9 10 \
        11 12 13 14 15 16 17 18 19 20 \
        21 22 23 24 25 26 27 28 29 30 \
    --batch-size 128 \
    --batches 4 \
    --seed 584001 \
    --device cuda \
    --out-dir "${WIDE_DIR}"

echo "long-J evaluation complete"
