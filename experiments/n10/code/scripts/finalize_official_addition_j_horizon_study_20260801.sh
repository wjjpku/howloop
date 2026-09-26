#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CONTROLLER_SEED=211001
EVALUATION_SEED=261001
OUT_DIR="${RUN_ROOT}/official_addition_j_horizon_aggregate"
SUMMARY="${OUT_DIR}/summary.json"

if [[ -f "${SUMMARY}" ]] && grep -q '"status": "complete"' "${SUMMARY}"; then
    echo "$(date -Is) addition J horizon aggregate already complete out_dir=${OUT_DIR}"
    exit 0
fi

while true; do
    COMPLETE=0
    for SEED in 0 1 2; do
        for LOGICAL_MAXIMUM in 19 40; do
            LABEL="addition_adaptive_step_official_seed${SEED}_rank48_logical1to${LOGICAL_MAXIMUM}_seed${CONTROLLER_SEED}"
            MANIFEST="${RUN_ROOT}/manifests/${LABEL}/audit.json"
            if [[ -f "${MANIFEST}" ]] && grep -q '"status": "complete"' "${MANIFEST}"; then
                COMPLETE=$((COMPLETE + 1))
            fi
        done
    done
    echo "$(date -Is) official_addition_J_horizon completed_audits=${COMPLETE}/6"
    if [[ "${COMPLETE}" -eq 6 ]]; then
        break
    fi
    sleep 60
done

cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.paper_length_horizon_aggregate \
    --task addition \
    --run-root "${RUN_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --seeds 0 1 2 \
    --logical-maxima 19 40 \
    --controller-seed "${CONTROLLER_SEED}" \
    --evaluation-seed "${EVALUATION_SEED}" \
    --lengths 19 25 30 40 50 60 75 100 \
    --minimum-examples 512

echo "$(date -Is) official addition J horizon aggregate complete out_dir=${OUT_DIR}"
