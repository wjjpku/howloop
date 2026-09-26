#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CONTROLLER_SEED=211001
EVALUATION_SEED=261001
OUT_DIR="${RUN_ROOT}/released64_j_horizon_aggregate"
SUMMARY="${OUT_DIR}/summary.json"

if [[ -f "${SUMMARY}" ]] && grep -q '"status": "complete"' "${SUMMARY}"; then
    echo "$(date -Is) J horizon aggregate already complete out_dir=${OUT_DIR}"
    exit 0
fi

while true; do
    COMPLETE=0
    RESOLVED_SEEDS=0
    for SEED in 0 1 2; do
        SEED_COMPLETE=0
        for LOGICAL_MAXIMUM in 20 40; do
            LABEL="parity_adaptive_step_released64_seed${SEED}_rank48_logical1to${LOGICAL_MAXIMUM}_seed${CONTROLLER_SEED}"
            MANIFEST="${RUN_ROOT}/manifests/${LABEL}/audit.json"
            if [[ -f "${MANIFEST}" ]] && grep -q '"status": "complete"' "${MANIFEST}"; then
                COMPLETE=$((COMPLETE + 1))
                SEED_COMPLETE=$((SEED_COMPLETE + 1))
            fi
        done
        DIAGNOSIS="${RUN_ROOT}/diagnosis/parity_adaptive_step_released64_seed${SEED}/summary.json"
        if [[ "${SEED_COMPLETE}" -eq 2 ]]; then
            RESOLVED_SEEDS=$((RESOLVED_SEEDS + 1))
        elif [[ -f "${DIAGNOSIS}" ]] \
            && grep -q '"status": "complete"' "${DIAGNOSIS}" \
            && grep -A12 '"disease_gate"' "${DIAGNOSIS}" | grep -q '"passed": false'; then
            RESOLVED_SEEDS=$((RESOLVED_SEEDS + 1))
        fi
    done
    echo "$(date -Is) released64_J_horizon resolved_seeds=${RESOLVED_SEEDS}/3 completed_audits=${COMPLETE}/6"
    if [[ "${RESOLVED_SEEDS}" -eq 3 ]]; then
        break
    fi
    sleep 60
done

cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.paper_length_horizon_aggregate \
    --run-root "${RUN_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --seeds 0 1 2 \
    --logical-maxima 20 40 \
    --controller-seed "${CONTROLLER_SEED}" \
    --evaluation-seed "${EVALUATION_SEED}" \
    --lengths 20 30 40 50 60 75 84 100 120 150 200 \
    --minimum-examples 512 \
    --allow-ineligible

echo "$(date -Is) released64 J horizon aggregate complete out_dir=${OUT_DIR}"
