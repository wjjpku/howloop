#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731
LOG_ROOT=/data/wujiaju/logs/graph_path_component_j_sweep_20260731
SCREEN_STATUS="${ROOT}/screen/STATUS.txt"
FOLLOWUP_STATUS="${ROOT}/FOLLOWUP_STATUS.txt"
CODE_DIR=/data/wujiaju/LooPlus_component_j_sweep_20260731

mkdir -p "${ROOT}" "${LOG_ROOT}"
printf "waiting_for_screen\n" > "${FOLLOWUP_STATUS}"
while [[ "$(cat "${SCREEN_STATUS}")" == "running" ]]; do
    sleep 20
done
if [[ "$(cat "${SCREEN_STATUS}")" != "complete" ]]; then
    printf "blocked_by_screen:%s\n" "$(cat "${SCREEN_STATUS}")" \
        > "${FOLLOWUP_STATUS}"
    exit 1
fi

printf "validating_screen\n" > "${FOLLOWUP_STATUS}"
bash "${CODE_DIR}/scripts/run_component_j_validation_remote.sh" \
    > "${LOG_ROOT}/validation_master.log" 2>&1

printf "running_combinations\n" > "${FOLLOWUP_STATUS}"
bash "${CODE_DIR}/scripts/run_component_j_combinations_remote.sh" \
    > "${LOG_ROOT}/combinations_master.log" 2>&1

printf "running_lowrank_refinement\n" > "${FOLLOWUP_STATUS}"
bash "${CODE_DIR}/scripts/run_component_j_lowrank_refinement_remote.sh" \
    > "${LOG_ROOT}/lowrank_refinement_master.log" 2>&1

printf "complete\n" > "${FOLLOWUP_STATUS}"
