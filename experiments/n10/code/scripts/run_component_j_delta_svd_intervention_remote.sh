#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/paperexperiment/LooPlus_prenorm_component_20260731
OUTPUT_DIR=/data/paperexperiment/graph_path_component_j_mechanism_20260731/full64_delta_svd
LOG_DIR=/data/paperexperiment/logs/graph_path_component_j_mechanism_20260731
LOG_FILE="${LOG_DIR}/full64_delta_svd.log"
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
BASE_J=/data/paperexperiment/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/h64/unit_j_maps.pt
TARGET_J=/data/paperexperiment/graph_path_component_j_sweep_20260731/combinations/full_lr1e6_h64/unit_j_maps.pt
TRAINING_STREAMS="${CODE_DIR}/training_streams_full_combo.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
printf "running\n" > "${OUTPUT_DIR}/STATUS.txt"
printf "%s\n" "$$" > "${OUTPUT_DIR}/REMOTE_SHELL_PID.txt"
date --iso-8601=seconds > "${OUTPUT_DIR}/STARTED_AT.txt"
cd "${CODE_DIR}"

set +e
/usr/bin/time -v "${PYTHON_BIN}" \
    -m reasoning_loop.graph_path_telomere_j_delta_svd_intervention \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --base-j-artifact "${BASE_J}" \
    --target-j-artifact "${TARGET_J}" \
    --out-dir "${OUTPUT_DIR}" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 64 \
    --continuation-loops 64 \
    --ranks 0 1 2 4 8 12 16 24 32 48 64 96 128 192 256 \
    --random-ranks 8 16 32 64 \
    --random-seeds 314159 271828 161803 \
    --geometry-cycles 1 8 16 24 32 48 64 \
    --partitions unseen \
    --training-streams-json "${TRAINING_STREAMS}" \
    > "${LOG_FILE}" 2>&1
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]] && [[ -s "${OUTPUT_DIR}/summary.json" ]]; then
    printf "complete\n" > "${OUTPUT_DIR}/STATUS.txt"
else
    printf "failed:%s\n" "${exit_code}" > "${OUTPUT_DIR}/STATUS.txt"
fi
date --iso-8601=seconds > "${OUTPUT_DIR}/FINISHED_AT.txt"
exit "${exit_code}"
