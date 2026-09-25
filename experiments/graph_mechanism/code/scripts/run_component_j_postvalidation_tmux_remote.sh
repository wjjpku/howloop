#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731
LOG_ROOT=/data/wujiaju/logs/graph_path_component_j_sweep_20260731
FOLLOWUP_STATUS="${ROOT}/FOLLOWUP_STATUS.txt"
STATUS="${ROOT}/POSTVALIDATION_STATUS.txt"
CODE_DIR=/data/wujiaju/LooPlus_component_j_sweep_20260731

printf "waiting_for_followups\n" > "${STATUS}"
while [[ "$(cat "${FOLLOWUP_STATUS}")" != "complete" ]]; do
    if [[ "$(cat "${FOLLOWUP_STATUS}")" == blocked_by_screen:* ]]; then
        printf "blocked:%s\n" "$(cat "${FOLLOWUP_STATUS}")" > "${STATUS}"
        exit 1
    fi
    sleep 20
done

printf "validating_combinations\n" > "${STATUS}"
bash "${CODE_DIR}/scripts/run_component_j_stage_validation_remote.sh" \
    combinations 5 > "${LOG_ROOT}/validation_combinations_master.log" 2>&1

printf "validating_lowrank_refinement\n" > "${STATUS}"
bash "${CODE_DIR}/scripts/run_component_j_stage_validation_remote.sh" \
    lowrank_refinement 8 \
    > "${LOG_ROOT}/validation_lowrank_refinement_master.log" 2>&1

printf "complete\n" > "${STATUS}"
