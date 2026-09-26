#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 TASK_SEED PHYSICAL_GPU" >&2
    exit 2
fi

TASK_SEED="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/paperexperiment/LooPlus_prenorm_component_20260731
ROOT=/data/paperexperiment/graph_path_component_j_ce_only_20260731
LOG_ROOT=/data/paperexperiment/logs/graph_path_component_j_ce_only_20260731
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
TARGET_J="${ROOT}/task${TASK_SEED}/ce_only_h64/unit_j_maps.pt"
TRAINING_STREAMS="${ROOT}/training_streams_task${TASK_SEED}.json"
OUTPUT_DIR="${ROOT}/paired_audit/task${TASK_SEED}"
LOG_FILE="${LOG_ROOT}/paired_audit_task${TASK_SEED}.log"

case "${TASK_SEED}" in
    404003)
        BASE_J=/data/paperexperiment/graph_path_component_j_sweep_20260731/combinations/full_lr1e6_h64/unit_j_maps.pt
        ;;
    411003|421003|431003)
        BASE_J="/data/paperexperiment/graph_path_component_j_sweep_20260731/multiseed/task${TASK_SEED}/full_lr1e6_h64/unit_j_maps.pt"
        ;;
    *)
        echo "unsupported matched task seed: ${TASK_SEED}" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
cd "${CODE_DIR}"
exec "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_j_delta_svd_intervention \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --base-j-artifact "${BASE_J}" \
    --target-j-artifact "${TARGET_J}" \
    --out-dir "${OUTPUT_DIR}" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 64 \
    --continuation-loops 64 \
    --sample-seed 512004 \
    --ranks 0 \
    --random-ranks 256 \
    --random-seeds 314159 \
    --geometry-cycles 1 8 16 24 32 48 64 \
    --partitions unseen \
    --training-streams-json "${TRAINING_STREAMS}" \
    > "${LOG_FILE}" 2>&1
