#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 LR_MULTIPLIER PHYSICAL_GPU" >&2
    exit 2
fi

LR_MULTIPLIER="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
OUTPUT_ROOT=/data/paperexperiment/graph_path_telomere_task_lora_j_ceonly_20260731
LOG_ROOT=/data/paperexperiment/logs
LABEL="svd_lr${LR_MULTIPLIER}"
OUT_DIR="${OUTPUT_ROOT}/pilot_boundary_${LABEL}"
RUN_LOG="${LOG_ROOT}/task_lora_pilot_boundary_${LABEL}.log"
HEARTBEAT_LOG="${LOG_ROOT}/task_lora_pilot_boundary_${LABEL}.heartbeat.log"
MANIFEST="${OUT_DIR}/run_manifest.json"
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/paperexperiment/graph_path_telomere_overloop_20260729/phase_grid/D8_L8_seed0/summary.json
REFERENCE_AFFINE=/data/paperexperiment/graph_path_telomere_unit_j_20260731/seed0_curriculum64/unit_j_maps.pt
INITIAL_AFFINE=/data/paperexperiment/graph_path_telomere_unit_j_20260731/seed0_focus_long/unit_j_maps.pt

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
printf '{\n  "status": "launching",\n  "placement": "loop_boundary",\n  "learning_rate_multiplier": %s,\n  "ranks": [8, 32, 128],\n  "initialization_seeds": [211001],\n  "physical_gpu": %s\n}\n' \
    "${LR_MULTIPLIER}" "${PHYSICAL_GPU}" > "${MANIFEST}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_lora_j \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --reference-affine-artifact "${REFERENCE_AFFINE}" \
    --reference-affine-label task \
    --initialization affine_svd \
    --initial-affine-artifact "${INITIAL_AFFINE}" \
    --initial-affine-label reg \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --placement loop_boundary \
    --ranks 8 32 128 \
    --initialization-seeds 211001 \
    --learning-rate-multiplier "${LR_MULTIPLIER}" \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --evaluation-loops 64 \
    --evaluation-seed 212004 \
    --cuda-memory-fraction 0.04 \
    --physical-gpu "${PHYSICAL_GPU}" \
    --prelaunch-used-mib "${PRELAUNCH_USED_MIB}" \
    --prelaunch-free-mib "${PRELAUNCH_FREE_MIB}" \
    --declared-peak-gib 3.0 \
    --reserve-gib 16.0 \
    > "${RUN_LOG}" 2>&1 &
TRAIN_PID=$!

printf '%s pid=%s lr=%s gpu=%s status=running\n' \
    "$(date -Is)" "${TRAIN_PID}" "${LR_MULTIPLIER}" "${PHYSICAL_GPU}" >> "${HEARTBEAT_LOG}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    printf '%s pid=%s lr=%s gpu=%s status=running\n' \
        "$(date -Is)" "${TRAIN_PID}" "${LR_MULTIPLIER}" "${PHYSICAL_GPU}" >> "${HEARTBEAT_LOG}"
    sleep 30
done

set +e
wait "${TRAIN_PID}"
STATUS=$?
set -e
if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
printf '%s pid=%s lr=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${TRAIN_PID}" "${LR_MULTIPLIER}" "${PHYSICAL_GPU}" \
    "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_LOG}"
printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "placement": "loop_boundary",\n  "learning_rate_multiplier": %s,\n  "ranks": [8, 32, 128],\n  "initialization_seeds": [211001],\n  "physical_gpu": %s\n}\n' \
    "${RUN_STATUS}" "${STATUS}" "${LR_MULTIPLIER}" "${PHYSICAL_GPU}" > "${MANIFEST}"
exit "${STATUS}"
