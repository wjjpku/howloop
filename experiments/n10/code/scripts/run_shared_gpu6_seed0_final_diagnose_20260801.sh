#!/usr/bin/env bash
set -euo pipefail

GPU=6
GPU_UUID=GPU-26e476d0-8829-ba32-ffe6-f482deac929e
EXPECTED_PIDS=(2531269 2712294)
RESERVE_MIB=16384
# The matching released64 FP32 run reserved 0.7754 GiB.  Use a more
# conservative declared bound for batch-16 no-grad streaming diagnosis.
DECLARED_PEAK_MIB=1536
REQUIRED_FREE_MIB=$((RESERVE_MIB + DECLARED_PEAK_MIB))
MEMORY_TOLERANCE_MIB=128

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
LABEL=parity_adaptive_step_released64_seed0
CHECKPOINT="${RUN_ROOT}/backbones/${LABEL}/final.pt"
OUT_DIR="${RUN_ROOT}/diagnosis/${LABEL}"
MANIFEST_DIR="${RUN_ROOT}/manifests/${LABEL}"
MANIFEST="${MANIFEST_DIR}/diagnose.json"
LOG="${LOG_ROOT}/${LABEL}_diagnose_shared_gpu6.log"
HEARTBEAT="${LOG_ROOT}/${LABEL}_diagnose_shared_gpu6.heartbeat.log"

mkdir -p "${OUT_DIR}" "${MANIFEST_DIR}" "${LOG_ROOT}"

gpu_field() {
    local field="$1"
    nvidia-smi -i "${GPU}" --query-gpu="${field}" \
        --format=csv,noheader,nounits | tr -d ' '
}

current_pid_memory() {
    nvidia-smi -i "${GPU}" --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

require_expected_process_set() {
    local rows="$1"
    local observed
    observed="$(printf '%s\n' "${rows}" | awk -F, 'NF {print $1}' | sort -n | xargs)"
    local expected
    expected="$(printf '%s\n' "${EXPECTED_PIDS[@]}" | sort -n | xargs)"
    if [[ "${observed}" != "${expected}" ]]; then
        echo "unexpected GPU process set: observed=${observed} expected=${expected}" >&2
        exit 75
    fi
}

if [[ "$(gpu_field uuid)" != "${GPU_UUID}" ]]; then
    echo "GPU UUID changed for physical GPU ${GPU}" >&2
    exit 75
fi

OBS1_TIME="$(date -Is)"
OBS1_ROWS="$(current_pid_memory)"
OBS1_FREE="$(gpu_field memory.free)"
OBS1_UTIL="$(gpu_field utilization.gpu)"
require_expected_process_set "${OBS1_ROWS}"
if (( OBS1_UTIL > 5 )); then
    echo "GPU ${GPU} is active at first observation: ${OBS1_UTIL}%" >&2
    exit 75
fi

sleep 61

OBS2_TIME="$(date -Is)"
OBS2_ROWS="$(current_pid_memory)"
OBS2_FREE="$(gpu_field memory.free)"
OBS2_UTIL="$(gpu_field utilization.gpu)"
require_expected_process_set "${OBS2_ROWS}"
if (( OBS2_UTIL > 5 )); then
    echo "GPU ${GPU} is active at second observation: ${OBS2_UTIL}%" >&2
    exit 75
fi
for pid in "${EXPECTED_PIDS[@]}"; do
    first="$(printf '%s\n' "${OBS1_ROWS}" | awk -F, -v p="${pid}" '$1==p {print $2}')"
    second="$(printf '%s\n' "${OBS2_ROWS}" | awk -F, -v p="${pid}" '$1==p {print $2}')"
    delta=$((second - first))
    if (( delta > MEMORY_TOLERANCE_MIB || delta < -MEMORY_TOLERANCE_MIB )); then
        echo "pre-existing PID ${pid} changed memory by ${delta} MiB" >&2
        exit 75
    fi
done
if (( OBS2_FREE < REQUIRED_FREE_MIB )); then
    echo "insufficient free memory: ${OBS2_FREE} < ${REQUIRED_FREE_MIB} MiB" >&2
    exit 75
fi

printf '{\n  "status": "launching",\n  "action": "diagnose",\n  "seed": 0,\n  "physical_gpu": 6,\n  "gpu_uuid": "%s",\n  "shared_gpu": true,\n  "observation_1": "%s",\n  "observation_2": "%s",\n  "preexisting_pid_memory_1": "%s",\n  "preexisting_pid_memory_2": "%s",\n  "prelaunch_free_mib": %s,\n  "declared_peak_mib": %s,\n  "reserve_mib": %s,\n  "log": "%s"\n}\n' \
    "${GPU_UUID}" "${OBS1_TIME}" "${OBS2_TIME}" \
    "$(printf '%s' "${OBS1_ROWS}" | tr '\n' ';')" \
    "$(printf '%s' "${OBS2_ROWS}" | tr '\n' ';')" \
    "${OBS2_FREE}" "${DECLARED_PEAK_MIB}" "${RESERVE_MIB}" "${LOG}" \
    > "${MANIFEST}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

"${PYTHON_BIN}" -u -m reasoning_loop.paper_length_telomere diagnose \
    --checkpoint "${CHECKPOINT}" \
    --batch-size 16 --batches 256 \
    --lengths 20 40 50 75 100 \
    --maximum-step 132 --seed 260001 \
    --extension-gate-length 100 --device cuda --out-dir "${OUT_DIR}" \
    > "${LOG}" 2>&1 &
RUN_PID=$!

retreat_reason=""
while kill -0 "${RUN_PID}" 2>/dev/null; do
    sleep 30
    free_mib="$(gpu_field memory.free)"
    rows="$(current_pid_memory)"
    if (( free_mib < RESERVE_MIB )); then
        retreat_reason="free_memory_below_reserve:${free_mib}"
    fi
    for pid in "${EXPECTED_PIDS[@]}"; do
        first="$(printf '%s\n' "${OBS2_ROWS}" | awk -F, -v p="${pid}" '$1==p {print $2}')"
        now="$(printf '%s\n' "${rows}" | awk -F, -v p="${pid}" '$1==p {print $2}')"
        if [[ -n "${now}" ]] && (( now > first + MEMORY_TOLERANCE_MIB )); then
            retreat_reason="preexisting_memory_growth:${pid}:${first}:${now}"
        fi
    done
    while IFS=, read -r pid _; do
        [[ -z "${pid}" ]] && continue
        allowed=false
        [[ "${pid}" == "${RUN_PID}" ]] && allowed=true
        for expected in "${EXPECTED_PIDS[@]}"; do
            [[ "${pid}" == "${expected}" ]] && allowed=true
        done
        if [[ "${allowed}" != true ]]; then
            retreat_reason="new_colocated_pid:${pid}"
        fi
    done <<< "${rows}"
    if grep -Eqi 'CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED' "${LOG}"; then
        retreat_reason="cuda_oom"
    fi
    printf '%s pid=%s free_mib=%s status=%s\n' \
        "$(date -Is)" "${RUN_PID}" "${free_mib}" \
        "${retreat_reason:-running}" >> "${HEARTBEAT}"
    if [[ -n "${retreat_reason}" ]]; then
        kill -TERM "${RUN_PID}" 2>/dev/null || true
        break
    fi
done

set +e
wait "${RUN_PID}"
exit_code=$?
set -e
if [[ -n "${retreat_reason}" ]]; then
    status=retreated
elif (( exit_code == 0 )); then
    status=complete
else
    status=failed
fi

printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "pid": %s,\n  "action": "diagnose",\n  "seed": 0,\n  "physical_gpu": 6,\n  "gpu_uuid": "%s",\n  "shared_gpu": true,\n  "observation_1": "%s",\n  "observation_2": "%s",\n  "preexisting_pid_memory_1": "%s",\n  "preexisting_pid_memory_2": "%s",\n  "prelaunch_free_mib": %s,\n  "declared_peak_mib": %s,\n  "reserve_mib": %s,\n  "retreat_reason": "%s",\n  "log": "%s"\n}\n' \
    "${status}" "${exit_code}" "${RUN_PID}" "${GPU_UUID}" \
    "${OBS1_TIME}" "${OBS2_TIME}" \
    "$(printf '%s' "${OBS1_ROWS}" | tr '\n' ';')" \
    "$(printf '%s' "${OBS2_ROWS}" | tr '\n' ';')" \
    "${OBS2_FREE}" "${DECLARED_PEAK_MIB}" "${RESERVE_MIB}" \
    "${retreat_reason}" "${LOG}" > "${MANIFEST}"

exit "${exit_code}"
