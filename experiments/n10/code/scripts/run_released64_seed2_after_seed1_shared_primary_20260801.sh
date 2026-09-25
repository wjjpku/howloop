#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
SEED1_FINAL_AUDIT="${RUN_ROOT}/manifests/parity_adaptive_step_released64_seed1_rank48_logical1to40_seed211001/audit.json"
PIPELINE=/data/wujiaju/LooPlus/scripts/run_released64_postbackbone_shared_primary_20260801.sh

while ! [[ -f "${SEED1_FINAL_AUDIT}" ]] \
    || ! grep -q '"status": "complete"' "${SEED1_FINAL_AUDIT}"; do
    echo "$(date -Is) waiting for seed-1 J40 audit before seed-2 shared launch"
    sleep 60
done

bash "${PIPELINE}" 2 6 --launch-backbone
