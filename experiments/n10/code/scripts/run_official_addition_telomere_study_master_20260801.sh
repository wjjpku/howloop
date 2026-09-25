#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
BASELINE_MANAGER=/data/wujiaju/LooPlus/scripts/run_official_task_baselines_20260801.sh
PIPELINE=/data/wujiaju/LooPlus/scripts/run_official_addition_telomere_seed_pipeline_20260801.sh

# Baseline training is independent of parity's J eligibility and aggregate.
# This serializes with the all-task baseline master through a task-level lock.
bash "${BASELINE_MANAGER}" addition

for SEED in 0 1 2; do
    SESSION="paper_tel_official_addition_seed${SEED}_pipeline"
    LOG="${RUN_ROOT}/queue/official_addition_seed${SEED}_pipeline.log"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" \
            "bash ${PIPELINE} ${SEED} >> ${LOG} 2>&1"
    fi
done

echo "$(date -Is) official addition baselines verified; post-backbone pipelines dispatched"
