#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 HIDDEN_WIDTH PHYSICAL_GPU" >&2
    exit 2
fi

HIDDEN_WIDTH="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
OUTPUT_ROOT=/data/wujiaju/graph_path_telomere_task_mlp_j_ceonly_20260731
LOG_ROOT=/data/wujiaju/logs
CHECKPOINT=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/wujiaju/graph_path_telomere_overloop_20260729/phase_grid/D8_L8_seed0/summary.json
POST_DAGGER_AFFINE=/data/wujiaju/graph_path_telomere_unit_j_20260731/seed0_focus_long/unit_j_maps.pt
REFERENCE_AFFINE=/data/wujiaju/graph_path_telomere_unit_j_20260731/seed0_curriculum64/unit_j_maps.pt
OUT_DIR="${OUTPUT_ROOT}/w${HIDDEN_WIDTH}"
RUN_LOG="${LOG_ROOT}/task_mlp_ceonly_w${HIDDEN_WIDTH}.log"
HEARTBEAT_LOG="${LOG_ROOT}/task_mlp_ceonly_w${HIDDEN_WIDTH}.heartbeat.log"
MANIFEST="${OUT_DIR}/run_manifest.json"

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
printf '{\n  "status": "launching",\n  "hidden_width": %s,\n  "physical_gpu": %s,\n  "state_loss_weight": 0.0,\n  "checkpoint": "%s",\n  "output_dir": "%s"\n}\n' \
    "${HIDDEN_WIDTH}" "${PHYSICAL_GPU}" "${CHECKPOINT}" "${OUT_DIR}" > "${MANIFEST}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_mlp_j \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --post-dagger-affine-artifact "${POST_DAGGER_AFFINE}" \
    --post-dagger-affine-label reg \
    --reference-affine-artifact "${REFERENCE_AFFINE}" \
    --reference-affine-label task \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --hidden-widths "${HIDDEN_WIDTH}" \
    --initialization-seeds 211001 311001 411001 \
    --state-loss-weight 0.0 \
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

printf '%s pid=%s width=%s gpu=%s status=running\n' \
    "$(date -Is)" "${TRAIN_PID}" "${HIDDEN_WIDTH}" "${PHYSICAL_GPU}" \
    >> "${HEARTBEAT_LOG}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    printf '%s pid=%s width=%s gpu=%s status=running\n' \
        "$(date -Is)" "${TRAIN_PID}" "${HIDDEN_WIDTH}" "${PHYSICAL_GPU}" \
        >> "${HEARTBEAT_LOG}"
    sleep 30
done

set +e
wait "${TRAIN_PID}"
STATUS=$?
set -e
if [[ "${STATUS}" -eq 0 ]]; then
    RUN_STATUS=complete
else
    RUN_STATUS=failed
fi
printf '%s pid=%s width=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${TRAIN_PID}" "${HIDDEN_WIDTH}" "${PHYSICAL_GPU}" \
    "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_LOG}"
printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "hidden_width": %s,\n  "physical_gpu": %s,\n  "state_loss_weight": 0.0,\n  "checkpoint": "%s",\n  "output_dir": "%s"\n}\n' \
    "${RUN_STATUS}" "${STATUS}" "${HIDDEN_WIDTH}" "${PHYSICAL_GPU}" \
    "${CHECKPOINT}" "${OUT_DIR}" > "${MANIFEST}"
exit "${STATUS}"
