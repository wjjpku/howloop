#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 PHYSICAL_GPU" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
OUT_DIR=/data/wujiaju/graph_path_telomere_explicit_diagonal_rrr_20260731/boundary_1024
LOG_ROOT=/data/wujiaju/logs
RUN_LOG="${LOG_ROOT}/explicit_diagonal_rrr_boundary_1024.log"
CHECKPOINT=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/wujiaju/graph_path_telomere_overloop_20260729/phase_grid/D8_L8_seed0/summary.json

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_explicit_diagonal_rrr \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --graphs 1024 \
    --batch-size 32 \
    --data-seed 20260731 \
    --answer-weight 28 \
    --identity-weight 1 \
    --ridge 0.01 \
    --iterations 12 \
    --diagonal-init dense_diagonal \
    --ranks 8 16 32 48 64 96 128 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --evaluation-loops 64 \
    --evaluation-seed 212004 \
    > "${RUN_LOG}" 2>&1
