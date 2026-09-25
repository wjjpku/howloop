#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus_prenorm_component_20260731
OUTPUT_DIR=/data/wujiaju/graph_path_prenorm_component_unit_j_curriculum64_20260731
LOG_DIR=/data/wujiaju/logs
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/wujiaju/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"
TRAINING_STREAMS="${CODE_DIR}/training_streams.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.04
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
date --iso-8601=seconds > "${OUTPUT_DIR}/RUN_STARTED_AT.txt"
echo running > "${OUTPUT_DIR}/STATUS.txt"
cd "${CODE_DIR}"

run_stage() {
    local stage_name="$1"
    shift
    echo "${stage_name}" > "${OUTPUT_DIR}/CURRENT_STAGE.txt"
    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_unit_j "$@" \
        > "${LOG_DIR}/graph_path_prenorm_component_unit_j_${stage_name}_20260731.log" 2>&1
    test -s "${OUTPUT_DIR}/${stage_name}/summary.json"
    test -s "${OUTPUT_DIR}/${stage_name}/unit_j_maps.pt"
}

COMMON_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --phase-summary "${PHASE_SUMMARY}"
    --device cuda
    --frozen-model-loss-placement
    "final CE at loop 8 plus intermediate CE on p_min(2t,D), t=1..7"
    --policies unit_every
    --train-start-ages 8
    --power-rounds 0
    --position-group all
    --operating-age 7
    --answer-weight 28
    --identity-weight 1
    --ridge 0.01
    --physical-gpu 2
    --prelaunch-used-mib 4
    --prelaunch-free-mib 81150
    --declared-peak-gib 3
    --reserve-gib 16
)

run_stage h24 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUTPUT_DIR}/h24" \
    --calibration-batch-size 256 \
    --calibration-batches 4 \
    --dagger-batch-size 128 \
    --dagger-batches-per-round 8 \
    --dagger-rounds 8 \
    --rollout-horizons 4 8 16 24 \
    --task-batch-size 32 \
    --task-batches-per-round 16 \
    --task-rounds 32 \
    --task-learning-rate 1e-5 \
    --task-state-loss-weight 0.1 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --continuation-loops 64 \
    --calibration-seed 281001 \
    --dagger-seed 281002 \
    --task-seed 281003 \
    --evaluation-seed 281004

run_stage h32 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUTPUT_DIR}/h32" \
    --initial-map-artifact "${OUTPUT_DIR}/h24/unit_j_maps.pt" \
    --initial-map-label task \
    --calibration-batch-size 16 \
    --calibration-batches 1 \
    --dagger-rounds 0 \
    --rollout-horizons 16 24 32 \
    --task-batch-size 32 \
    --task-batches-per-round 9 \
    --task-rounds 16 \
    --task-learning-rate 3e-6 \
    --task-state-loss-weight 0.1 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --continuation-loops 48 \
    --calibration-seed 282001 \
    --task-seed 282003 \
    --evaluation-seed 282004

run_stage h48 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUTPUT_DIR}/h48" \
    --initial-map-artifact "${OUTPUT_DIR}/h32/unit_j_maps.pt" \
    --initial-map-label task \
    --calibration-batch-size 16 \
    --calibration-batches 1 \
    --dagger-rounds 0 \
    --rollout-horizons 24 32 48 \
    --task-batch-size 32 \
    --task-batches-per-round 9 \
    --task-rounds 16 \
    --task-learning-rate 3e-6 \
    --task-state-loss-weight 0.1 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --continuation-loops 64 \
    --calibration-seed 283001 \
    --task-seed 283003 \
    --evaluation-seed 283004

run_stage h64 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUTPUT_DIR}/h64" \
    --initial-map-artifact "${OUTPUT_DIR}/h48/unit_j_maps.pt" \
    --initial-map-label task \
    --calibration-batch-size 16 \
    --calibration-batches 1 \
    --dagger-rounds 0 \
    --rollout-horizons 32 48 64 \
    --task-batch-size 32 \
    --task-batches-per-round 9 \
    --task-rounds 16 \
    --task-learning-rate 3e-6 \
    --task-state-loss-weight 0.1 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --continuation-loops 96 \
    --calibration-seed 284001 \
    --task-seed 284003 \
    --evaluation-seed 284004

run_stage formal_eval \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUTPUT_DIR}/formal_eval" \
    --initial-map-artifact "${OUTPUT_DIR}/h64/unit_j_maps.pt" \
    --initial-map-label task \
    --calibration-batch-size 16 \
    --calibration-batches 1 \
    --dagger-rounds 0 \
    --rollout-horizons 32 48 64 \
    --task-rounds 0 \
    --evaluation-batch-size 128 \
    --evaluation-batches 16 \
    --continuation-loops 96 \
    --calibration-seed 285001 \
    --evaluation-seed 285004

echo strict_unseen_audit > "${OUTPUT_DIR}/CURRENT_STAGE.txt"
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_graph_leakage_audit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --j-artifact "${OUTPUT_DIR}/h64/unit_j_maps.pt" \
    --j-label task \
    --out-dir "${OUTPUT_DIR}/strict_unseen_audit" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 128 \
    --continuation-loops 64 \
    --sample-seed 286004 \
    --training-streams-json "${TRAINING_STREAMS}" \
    > "${LOG_DIR}/graph_path_prenorm_component_unit_j_strict_unseen_audit_20260731.log" 2>&1
test -s "${OUTPUT_DIR}/strict_unseen_audit/summary.json"

echo complete > "${OUTPUT_DIR}/STATUS.txt"
date --iso-8601=seconds > "${OUTPUT_DIR}/RUN_FINISHED_AT.txt"
