#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus_component_j_sweep_20260731
OUTPUT_ROOT=/data/wujiaju/graph_path_component_j_sweep_20260731/lowrank_refinement
LOG_ROOT=/data/wujiaju/logs/graph_path_component_j_sweep_20260731/lowrank_refinement
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/wujiaju/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
INITIAL_J=/data/wujiaju/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/h64/unit_j_maps.pt
CONFIGS="${CODE_DIR}/scripts/component_j_lowrank_refinement_configs.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${CODE_DIR}"
exec "${PYTHON_BIN}" scripts/graph_path_component_j_sweep.py \
    --code-dir "${CODE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --initial-j "${INITIAL_J}" \
    --output-root "${OUTPUT_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --configs-json "${CONFIGS}" \
    --task-seed 405003 \
    --evaluation-seed 405004 \
    --physical-gpu 2 \
    --prelaunch-used-mib 4 \
    --prelaunch-free-mib 81916
