#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/paperexperiment/LooPlus_component_j_sweep_20260731
ROOT=/data/paperexperiment/graph_path_component_j_sweep_20260731
OUTPUT_ROOT="${ROOT}/strict_unseen"
LOG_ROOT=/data/paperexperiment/logs/graph_path_component_j_sweep_20260731/strict_unseen
STATUS="${ROOT}/STRICT_UNSEEN_STATUS.txt"
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
STREAM_ROOT="${CODE_DIR}/results/graph_path_component_j_sweep_20260731"
WAIT_STATUS="${ROOT}/LONG_VALIDATION_STATUS.txt"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
while [[ "$(cat "${WAIT_STATUS}")" != "complete" ]]; do
    sleep 20
done
printf "running\n" > "${STATUS}"

for candidate in \
    full64:64:${ROOT}/combinations/full_lr1e6_h64/unit_j_maps.pt:${STREAM_ROOT}/training_streams_full_combo.json \
    lora64:64:${ROOT}/lowrank_refinement/lora_r64_lr1e6_h64/unit_j_maps.pt:${STREAM_ROOT}/training_streams_lora_r64.json \
    extend_h64_80_96:96:${ROOT}/long_horizon/extend_lr5e7_h64_80_96/unit_j_maps.pt:${STREAM_ROOT}/training_streams_long_horizon.json \
    extend_state003_h96:96:${ROOT}/long_horizon/extend_lr5e7_state003_h96/unit_j_maps.pt:${STREAM_ROOT}/training_streams_long_horizon.json
do
    name=${candidate%%:*}
    remainder=${candidate#*:}
    loops=${remainder%%:*}
    remainder=${remainder#*:}
    artifact=${remainder%%:*}
    streams=${remainder#*:}
    printf "%s\n" "${name}" > "${STATUS}"
    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_graph_leakage_audit \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --j-artifact "${artifact}" \
        --j-label task \
        --out-dir "${OUTPUT_ROOT}/${name}" \
        --device cuda \
        --sample-per-partition 512 \
        --batch-size 128 \
        --continuation-loops "${loops}" \
        --sample-seed 412004 \
        --operating-age 3 \
        --training-streams-json "${streams}" \
        > "${LOG_ROOT}/${name}.log" 2>&1
done

printf "complete\n" > "${STATUS}"
