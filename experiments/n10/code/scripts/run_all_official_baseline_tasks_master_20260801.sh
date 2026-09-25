#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
MANAGER=/data/wujiaju/LooPlus/scripts/run_official_task_baselines_20260801.sh
mkdir -p "${RUN_ROOT}/queue"

# Parity seeds 0/1/2 are already complete.  The remaining three managers each
# run their seeds serially; the waiter enforces a global three-job GPU limit.
for TASK in copy addition sum_reverse; do
    SESSION="paper_tel_official_${TASK}_baselines"
    LOG="${RUN_ROOT}/queue/official_${TASK}_baselines.log"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" \
            "bash ${MANAGER} ${TASK} >> ${LOG} 2>&1"
    fi
done

echo "$(date -Is) official baseline managers dispatched tasks=copy,addition,sum_reverse seeds=0,1,2"
