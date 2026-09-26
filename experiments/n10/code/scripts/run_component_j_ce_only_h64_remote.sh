#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 TASK_SEED PHYSICAL_GPU" >&2
    exit 2
fi

TASK_SEED="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/paperexperiment/LooPlus_component_j_sweep_20260731
ROOT=/data/paperexperiment/graph_path_component_j_ce_only_20260731
LOG_ROOT=/data/paperexperiment/logs/graph_path_component_j_ce_only_20260731
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
INITIAL_J=/data/paperexperiment/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/h64/unit_j_maps.pt
CONFIGS="${CODE_DIR}/scripts/component_j_ce_only_h64_config.json"
OUTPUT_ROOT="${ROOT}/task${TASK_SEED}"
STAGE_LOG_ROOT="${LOG_ROOT}/task${TASK_SEED}"
PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${STAGE_LOG_ROOT}"
cd "${CODE_DIR}"
exec "${PYTHON_BIN}" scripts/graph_path_component_j_sweep.py \
    --code-dir "${CODE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --initial-j "${INITIAL_J}" \
    --output-root "${OUTPUT_ROOT}" \
    --log-root "${STAGE_LOG_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --configs-json "${CONFIGS}" \
    --task-seed "${TASK_SEED}" \
    --evaluation-seed 509004 \
    --physical-gpu "${PHYSICAL_GPU}" \
    --prelaunch-used-mib "${PRELAUNCH_USED_MIB}" \
    --prelaunch-free-mib "${PRELAUNCH_FREE_MIB}"
