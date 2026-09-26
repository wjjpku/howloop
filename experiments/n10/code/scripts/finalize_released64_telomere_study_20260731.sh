#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
CONTROLLER_SEED=211001
CURRICULUM=extension
OUT_DIR="${RUN_ROOT}/released64_formal_aggregate"
SUMMARIZER=/data/paperexperiment/LooPlus/scripts/summarize_released64_telomere_study.py
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python

while true; do
    COMPLETE=0
    for SEED in 0 1 2; do
        LABEL="parity_adaptive_step_released64_seed${SEED}_rank48_${CURRICULUM}_seed${CONTROLLER_SEED}"
        MANIFEST="${RUN_ROOT}/manifests/${LABEL}/audit.json"
        if [[ -f "${MANIFEST}" ]] && grep -q '"status": "complete"' "${MANIFEST}"; then
            COMPLETE=$((COMPLETE + 1))
        fi
    done
    echo "$(date -Is) released64 completed_audits=${COMPLETE}/3"
    if [[ "${COMPLETE}" -eq 3 ]]; then
        break
    fi
    sleep 60
done

"${PYTHON_BIN}" "${SUMMARIZER}" \
    --run-root "${RUN_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --seeds 0 1 2 \
    --curriculum "${CURRICULUM}" \
    --controller-seed "${CONTROLLER_SEED}"

echo "$(date -Is) released64 aggregate complete out_dir=${OUT_DIR}"
