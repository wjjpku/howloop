#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
PIPELINE=/data/paperexperiment/LooPlus/scripts/run_official_shared_postbackbone_seed_20260802.sh
mkdir -p "${RUN_ROOT}/queue"

# Task slots match the baseline scheduler: Copy=0, Addition=1, Sum-Reverse=2.
# Addition's manager can start immediately but its slot remains held by the
# still-running seed-2 backbone, so it cannot oversubscribe the third GPU job.
for SPEC in "copy 4" "addition 3" "sum_reverse 6"; do
    read -r TASK GPU <<< "${SPEC}"
    SESSION="paper_tel_official_${TASK}_postbackbone"
    LOG="${RUN_ROOT}/queue/official_${TASK}_postbackbone.log"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" \
            "for seed in 0 1 2; do bash ${PIPELINE} ${TASK} \${seed} ${GPU} || exit \$?; done >> ${LOG} 2>&1"
    fi
done

echo "$(date -Is) official post-backbone managers dispatched tasks=copy,addition,sum_reverse"
