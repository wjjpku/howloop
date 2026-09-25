#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus_component_j_sweep_20260731
ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731
LOG_ROOT=/data/wujiaju/logs/graph_path_component_j_sweep_20260731
WAIT_STATUS="${ROOT}/MULTISEED_STATUS.txt"
STATUS="${ROOT}/LONG_HORIZON_STATUS.txt"
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/wujiaju/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
INITIAL_J="${ROOT}/combinations/full_lr1e6_h64/unit_j_maps.pt"
CONFIGS="${CODE_DIR}/scripts/component_j_long_horizon_configs.json"
OUTPUT_ROOT="${ROOT}/long_horizon"
STAGE_LOG_ROOT="${LOG_ROOT}/long_horizon"

printf "waiting_for_multiseed\n" > "${STATUS}"
while [[ "$(cat "${WAIT_STATUS}")" != "complete" ]]; do
    sleep 20
done

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${STAGE_LOG_ROOT}"
printf "running\n" > "${STATUS}"
"${PYTHON_BIN}" "${CODE_DIR}/scripts/graph_path_component_j_sweep.py" \
    --code-dir "${CODE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --initial-j "${INITIAL_J}" \
    --output-root "${OUTPUT_ROOT}" \
    --log-root "${STAGE_LOG_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --configs-json "${CONFIGS}" \
    --task-seed 441003 \
    --evaluation-seed 409004 \
    --evaluation-batches 4 \
    --continuation-loops 128 \
    --physical-gpu 2 \
    --prelaunch-used-mib 4 \
    --prelaunch-free-mib 81916 \
    > "${STAGE_LOG_ROOT}/master.log" 2>&1
printf "complete\n" > "${STATUS}"
