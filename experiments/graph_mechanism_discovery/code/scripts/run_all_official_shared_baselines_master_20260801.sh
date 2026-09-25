#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
SHARED_MANAGER=/data/wujiaju/LooPlus/scripts/run_official_shared_task_baselines_20260801.sh
mkdir -p "${RUN_ROOT}/queue"

# Establish matching peak-memory measurements first.  Each one has a bounded
# allocator cap and retains at least 16 GiB for the pre-existing workload.
for SPEC in "copy 6" "addition 3" "sum_reverse 4"; do
    read -r TASK GPU <<< "${SPEC}"
    bash "${SHARED_MANAGER}" benchmark "${TASK}" "${GPU}"
done

# Then use at most three co-located cards.  GPU6 is compute-idle; GPUs 3 and 4
# have ample memory reserve but may yield noisy wall-clock timing.
for SPEC in "copy 6" "addition 3" "sum_reverse 4"; do
    read -r TASK GPU <<< "${SPEC}"
    SESSION="paper_tel_official_${TASK}_shared_baselines"
    LOG="${RUN_ROOT}/queue/official_${TASK}_shared_baselines.log"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" \
            "bash ${SHARED_MANAGER} formal ${TASK} ${GPU} >> ${LOG} 2>&1"
    fi
done

echo "$(date -Is) shared official formal managers dispatched"
