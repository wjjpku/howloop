#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/paperexperiment/LooPlus_prenorm_component_20260731
OUTPUT_DIR=/data/paperexperiment/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/schur_intervention/formal_eval
LOG_FILE=/data/paperexperiment/logs/graph_path_component_j_schur_intervention_formal_20260731.log
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
J_ARTIFACT=/data/paperexperiment/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/h64/unit_j_maps.pt
SCHUR_ARTIFACT=/data/paperexperiment/graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731/schur_intervention/real_schur_bands.pt
TRAINING_STREAMS="${CODE_DIR}/training_streams_direct_H3.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"
printf "running\n" > "${OUTPUT_DIR}/STATUS.txt"
printf "%s\n" "$$" > "${OUTPUT_DIR}/REMOTE_SHELL_PID.txt"
date --iso-8601=seconds > "${OUTPUT_DIR}/STARTED_AT.txt"
cd "${CODE_DIR}"

set +e
/usr/bin/time -v "${PYTHON_BIN}" \
    -m reasoning_loop.graph_path_telomere_j_schur_intervention \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --j-artifact "${J_ARTIFACT}" \
    --j-label task \
    --schur-artifact "${SCHUR_ARTIFACT}" \
    --out-dir "${OUTPUT_DIR}" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 128 \
    --continuation-loops 64 \
    --operating-age 3 \
    --sample-seed 296004 \
    --partitions seen unseen \
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
