#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus_component_j_sweep_20260731
ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731
LOG_ROOT=/data/wujiaju/logs/graph_path_component_j_sweep_20260731
WAIT_STATUS="${ROOT}/POSTVALIDATION_STATUS.txt"
STATUS="${ROOT}/MULTISEED_STATUS.txt"
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/wujiaju/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
INITIAL_J=/data/wujiaju/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/h64/unit_j_maps.pt
CONFIGS="${CODE_DIR}/scripts/component_j_multiseed_configs.json"

printf "waiting_for_postvalidation\n" > "${STATUS}"
while [[ "$(cat "${WAIT_STATUS}")" != "complete" ]]; do
    sleep 20
done

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for task_seed in 411003 421003 431003; do
    output_root="${ROOT}/multiseed/task${task_seed}"
    stage_log_root="${LOG_ROOT}/multiseed/task${task_seed}"
    mkdir -p "${output_root}" "${stage_log_root}"
    printf "running_task_seed:%s\n" "${task_seed}" > "${STATUS}"
    "${PYTHON_BIN}" "${CODE_DIR}/scripts/graph_path_component_j_sweep.py" \
        --code-dir "${CODE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --initial-j "${INITIAL_J}" \
        --output-root "${output_root}" \
        --log-root "${stage_log_root}" \
        --python-bin "${PYTHON_BIN}" \
        --configs-json "${CONFIGS}" \
        --task-seed "${task_seed}" \
        --evaluation-seed 408004 \
        --physical-gpu 2 \
        --prelaunch-used-mib 4 \
        --prelaunch-free-mib 81916 \
        > "${stage_log_root}/master.log" 2>&1
done

printf "complete\n" > "${STATUS}"
