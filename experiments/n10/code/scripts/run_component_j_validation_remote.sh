#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/paperexperiment/LooPlus_component_j_sweep_20260731
SCREEN_ROOT=/data/paperexperiment/graph_path_component_j_sweep_20260731/screen
OUTPUT_ROOT=/data/paperexperiment/graph_path_component_j_sweep_20260731/validation
LOG_ROOT=/data/paperexperiment/logs/graph_path_component_j_sweep_20260731/validation
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${CODE_DIR}"
exec "${PYTHON_BIN}" scripts/graph_path_component_j_validate.py \
    --code-dir "${CODE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --screen-root "${SCREEN_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --top-k 12 \
    --evaluation-seeds 402004 403004 \
    --evaluation-batches 8 \
    --physical-gpu 2
