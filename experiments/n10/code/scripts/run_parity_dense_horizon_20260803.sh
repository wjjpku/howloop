#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
OUT_DIR="${RUN_ROOT}/parity_far_horizon/anchor1_seed0_dense32_multires_l100to1000"
LOG_PATH="${LOG_ROOT}/parity_anchor1_seed0_dense32_multires_l100to1000.log"
HEARTBEAT_PATH="${LOG_ROOT}/parity_anchor1_seed0_dense32_multires_l100to1000.heartbeat.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
MEASURED_PEAK_MIB=256
RESERVE_MIB=16384

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
REQUIRED_FREE_MIB=$((MEASURED_PEAK_MIB + RESERVE_MIB))
if [[ "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "GPU ${PHYSICAL_GPU} has ${PRELAUNCH_FREE_MIB}MiB free; ${REQUIRED_FREE_MIB}MiB required" >&2
    exit 75
fi

LENGTHS=()
for ((length = 100; length < 300; length += 5)); do LENGTHS+=("${length}"); done
for ((length = 300; length < 600; length += 10)); do LENGTHS+=("${length}"); done
for ((length = 600; length <= 1000; length += 20)); do LENGTHS+=("${length}"); done

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.075
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
cd "${CODE_DIR}"

COMMAND=(
    "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py
    --checkpoint "${RUN_ROOT}/backbones/parity_adaptive_step_released64_seed0/final.pt"
    --controller "${RUN_ROOT}/controllers/parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001/controller.pt"
    --lengths "${LENGTHS[@]}"
    --examples 32
    --max-batch-size 32
    --token-budget 8192
    --seed 261001
    --device cuda
    --out-dir "${OUT_DIR}"
)

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" \
    "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]),
    "prelaunch_used_mib": int(sys.argv[3]),
    "prelaunch_free_mib": int(sys.argv[4]),
    "prelaunch_utilization_percent": int(sys.argv[5]),
    "preexisting_pids": sys.argv[6].split(),
    "measured_matching_peak_mib": 256,
    "reserve_mib": 16384,
    "examples_per_length": 32,
    "grid": {"100_295": 5, "300_590": 10, "600_1000": 20},
    "retained_intermediate_states": False,
}, indent=2, sort_keys=True) + "\n")
PY

"${COMMAND[@]}" > "${LOG_PATH}" 2>&1 &
CHILD_PID=$!
while kill -0 "${CHILD_PID}" 2>/dev/null; do
    FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s pid=%s gpu=%s status=running free_mib=%s utilization=%s\n' \
        "$(date -Is)" "${CHILD_PID}" "${PHYSICAL_GPU}" "${FREE_MIB}" "${UTILIZATION}" >> "${HEARTBEAT_PATH}"
    if [[ "${FREE_MIB}" -lt "${RESERVE_MIB}" ]]; then
        printf '%s pid=%s gpu=%s status=retreat_below_reserve free_mib=%s\n' \
            "$(date -Is)" "${CHILD_PID}" "${PHYSICAL_GPU}" "${FREE_MIB}" >> "${HEARTBEAT_PATH}"
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" || true
        exit 76
    fi
    sleep 30
done

set +e
wait "${CHILD_PID}"
STATUS=$?
set -e
if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${RUN_STATUS}" "${STATUS}" "${CHILD_PID}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "pid": int(sys.argv[4]),
    "finished_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
printf '%s pid=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${CHILD_PID}" "${PHYSICAL_GPU}" "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_PATH}"
exit "${STATUS}"
