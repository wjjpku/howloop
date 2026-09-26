#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/paperexperiment/LooPlus_component_j_sweep_20260731
ROOT=/data/paperexperiment/graph_path_component_j_sweep_20260731
OUTPUT_ROOT="${ROOT}/validation_long_horizon"
LOG_ROOT=/data/paperexperiment/logs/graph_path_component_j_sweep_20260731/validation_long_horizon
STATUS="${ROOT}/LONG_VALIDATION_STATUS.txt"
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
CHECKPOINT=/data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt
PHASE_SUMMARY="${CODE_DIR}/phase_summary_twohop.json"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${CODE_DIR}"
export TELOMERE_CUDA_MEMORY_FRACTION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
printf "running\n" > "${STATUS}"

for candidate in \
    base:/data/paperexperiment/graph_path_component_j_sweep_20260731/combinations/full_lr1e6_h64/unit_j_maps.pt \
    extend_h64_80_96:/data/paperexperiment/graph_path_component_j_sweep_20260731/long_horizon/extend_lr5e7_h64_80_96/unit_j_maps.pt \
    extend_state003_h96:/data/paperexperiment/graph_path_component_j_sweep_20260731/long_horizon/extend_lr5e7_state003_h96/unit_j_maps.pt
do
    name=${candidate%%:*}
    artifact=${candidate#*:}
    for seed in 410004 411004; do
        run_name="${name}_eval${seed}"
        printf "%s\n" "${run_name}" > "${STATUS}"
        "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_unit_j \
            --checkpoint "${CHECKPOINT}" \
            --phase-summary "${PHASE_SUMMARY}" \
            --out-dir "${OUTPUT_ROOT}/${run_name}" \
            --device cuda \
            --frozen-model-loss-placement \
            "final CE at loop 8 plus intermediate CE on p_min(2t,D), t=1..7" \
            --initial-map-artifact "${artifact}" \
            --initial-map-label task \
            --calibration-batch-size 16 \
            --calibration-batches 1 \
            --dagger-rounds 0 \
            --power-rounds 0 \
            --task-rounds 0 \
            --rollout-horizons 64 80 96 \
            --policies unit_every \
            --train-start-ages 8 \
            --position-group all \
            --operating-age 3 \
            --direct-operating-target-age 3 \
            --evaluation-batch-size 128 \
            --evaluation-batches 8 \
            --continuation-loops 128 \
            --calibration-seed 410001 \
            --evaluation-seed "${seed}" \
            --physical-gpu 2 \
            --declared-peak-gib 3 \
            --reserve-gib 16 \
            > "${LOG_ROOT}/${run_name}.log" 2>&1
    done
done

printf "complete\n" > "${STATUS}"
