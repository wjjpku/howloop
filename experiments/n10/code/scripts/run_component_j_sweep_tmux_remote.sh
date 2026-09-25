#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731/screen
MASTER_LOG=/data/wujiaju/logs/graph_path_component_j_sweep_20260731/master.log
RUNNER=/data/wujiaju/LooPlus_component_j_sweep_20260731/scripts/run_component_j_sweep_remote.sh

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${MASTER_LOG}")"
printf "running\n" > "${OUTPUT_ROOT}/STATUS.txt"
printf "%s\n" "$$" > "${OUTPUT_ROOT}/REMOTE_SHELL_PID.txt"
date --iso-8601=seconds > "${OUTPUT_ROOT}/STARTED_AT.txt"

set +e
bash "${RUNNER}" > "${MASTER_LOG}" 2>&1
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]] && [[ -s "${OUTPUT_ROOT}/ranking.json" ]]; then
    printf "complete\n" > "${OUTPUT_ROOT}/STATUS.txt"
else
    printf "failed:%s\n" "${exit_code}" > "${OUTPUT_ROOT}/STATUS.txt"
fi
date --iso-8601=seconds > "${OUTPUT_ROOT}/FINISHED_AT.txt"
exit "${exit_code}"
