#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 PLACEMENT PHYSICAL_GPU" >&2
    exit 2
fi

PLACEMENT="$1"
PHYSICAL_GPU="$2"
if [[ "${PLACEMENT}" != "pre_block2" && "${PLACEMENT}" != "loop_boundary" ]]; then
    echo "placement must be pre_block2 or loop_boundary" >&2
    exit 2
fi

CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/graph_path_telomere_task_lora_j_ceonly_20260731
OUT_DIR="${RUN_ROOT}/strict_unseen_${PLACEMENT}"
ARTIFACT="${RUN_ROOT}/${PLACEMENT}/task_lora_j.pt"
if [[ -f "${RUN_ROOT}/${PLACEMENT}/task_lora_j_stable.pt" ]]; then
    ARTIFACT="${RUN_ROOT}/${PLACEMENT}/task_lora_j_stable.pt"
fi
LOG_ROOT=/data/wujiaju/logs
RUN_LOG="${LOG_ROOT}/task_lora_strict_${PLACEMENT}.log"
HEARTBEAT_LOG="${LOG_ROOT}/task_lora_strict_${PLACEMENT}.heartbeat.log"
MANIFEST="${OUT_DIR}/run_manifest.json"
CHECKPOINT=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/wujiaju/graph_path_telomere_overloop_20260729/phase_grid/D8_L8_seed0/summary.json
REFERENCE_AFFINE=/data/wujiaju/graph_path_telomere_unit_j_20260731/seed0_curriculum64/unit_j_maps.pt

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
printf '{\n  "status": "launching",\n  "placement": "%s",\n  "physical_gpu": %s,\n  "sample_per_partition": 512,\n  "artifact": "%s"\n}\n' \
    "${PLACEMENT}" "${PHYSICAL_GPU}" "${ARTIFACT}" > "${MANIFEST}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --affine-artifact "${REFERENCE_AFFINE}" \
    --affine-label task \
    --lora-artifacts "${ARTIFACT}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 32 \
    --continuation-loops 64 \
    --sample-seed 20260731 \
    --cuda-memory-fraction 0.12 \
    > "${RUN_LOG}" 2>&1 &
AUDIT_PID=$!

printf '%s pid=%s placement=%s gpu=%s status=running\n' \
    "$(date -Is)" "${AUDIT_PID}" "${PLACEMENT}" "${PHYSICAL_GPU}" >> "${HEARTBEAT_LOG}"
while kill -0 "${AUDIT_PID}" 2>/dev/null; do
    printf '%s pid=%s placement=%s gpu=%s status=running\n' \
        "$(date -Is)" "${AUDIT_PID}" "${PLACEMENT}" "${PHYSICAL_GPU}" >> "${HEARTBEAT_LOG}"
    sleep 30
done

set +e
wait "${AUDIT_PID}"
STATUS=$?
set -e
if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
printf '%s pid=%s placement=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${AUDIT_PID}" "${PLACEMENT}" "${PHYSICAL_GPU}" \
    "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_LOG}"
printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "placement": "%s",\n  "physical_gpu": %s,\n  "sample_per_partition": 512,\n  "artifact": "%s"\n}\n' \
    "${RUN_STATUS}" "${STATUS}" "${PLACEMENT}" "${PHYSICAL_GPU}" "${ARTIFACT}" > "${MANIFEST}"
exit "${STATUS}"
