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
OUT_ROOT="${RUN_ROOT}/dense_horizon_20260803"
MANIFEST_PATH="${OUT_ROOT}/addition_sumreverse_dense64_l19to100_manifest.json"
HEARTBEAT_PATH="${LOG_ROOT}/addition_sumreverse_dense64_l19to100.heartbeat.log"
MEASURED_MATCHING_PEAK_MIB=6144
RESERVE_MIB=16384

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
REQUIRED_FREE_MIB=$((MEASURED_MATCHING_PEAK_MIB + RESERVE_MIB))
if [[ "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "GPU ${PHYSICAL_GPU} has ${PRELAUNCH_FREE_MIB}MiB free; ${REQUIRED_FREE_MIB}MiB required" >&2
    exit 75
fi

LENGTHS=()
for ((length = 19; length <= 100; length += 1)); do LENGTHS+=("${length}"); done

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.075
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
cd "${CODE_DIR}"

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
    "measured_matching_peak_mib": 6144,
    "reserve_mib": 16384,
    "tasks": ["addition", "sum_reverse"],
    "examples_per_length": 64,
    "grid": {"minimum": 19, "maximum": 100, "step": 1},
    "retained_intermediate_states": False,
    "completed_tasks": [],
}, indent=2, sort_keys=True) + "\n")
PY

run_task() {
    local task="$1"
    local checkpoint="${RUN_ROOT}/backbones/${task}_adaptive_step_official_seed0/final.pt"
    local controller="${RUN_ROOT}/controllers/${task}_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_postfinal_wsd_lr1em4_warmup2048_stable2816_5k_seed211001/controller.pt"
    local out_dir="${OUT_ROOT}/${task}_seed0_best5k_dense64_l19to100"
    local log_path="${LOG_ROOT}/${task}_seed0_best5k_dense64_l19to100.log"
    mkdir -p "${out_dir}"

    local free_mib
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "GPU ${PHYSICAL_GPU} has ${free_mib}MiB free before ${task}; ${REQUIRED_FREE_MIB}MiB required" >&2
        return 75
    fi

    "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${task}" \
        --checkpoint "${checkpoint}" \
        --controller "${controller}" \
        --lengths "${LENGTHS[@]}" \
        --examples 64 \
        --max-batch-size 32 \
        --token-budget 8192 \
        --seed 271001 \
        --device cuda \
        --out-dir "${out_dir}" > "${log_path}" 2>&1 &
    local child_pid=$!

    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${task}" "${child_pid}" "${out_dir}" "${log_path}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "running",
    "active_task": sys.argv[2],
    "pid": int(sys.argv[3]),
    "active_out_dir": sys.argv[4],
    "active_log": sys.argv[5],
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

    while kill -0 "${child_pid}" 2>/dev/null; do
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        local utilization
        utilization="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        printf '%s task=%s pid=%s gpu=%s status=running free_mib=%s utilization=%s\n' \
            "$(date -Is)" "${task}" "${child_pid}" "${PHYSICAL_GPU}" "${free_mib}" "${utilization}" >> "${HEARTBEAT_PATH}"
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            printf '%s task=%s pid=%s gpu=%s status=retreat_below_reserve free_mib=%s\n' \
                "$(date -Is)" "${task}" "${child_pid}" "${PHYSICAL_GPU}" "${free_mib}" >> "${HEARTBEAT_PATH}"
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        sleep 30
    done

    set +e
    wait "${child_pid}"
    local task_status=$?
    set -e
    if [[ "${task_status}" -ne 0 ]]; then
        return "${task_status}"
    fi
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${task}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
completed = list(payload.get("completed_tasks", []))
completed.append(sys.argv[2])
payload.update({
    "completed_tasks": completed,
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

STATUS=0
for TASK in addition sum_reverse; do
    set +e
    run_task "${TASK}"
    STATUS=$?
    set -e
    if [[ "${STATUS}" -ne 0 ]]; then
        break
    fi
done

if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${RUN_STATUS}" "${STATUS}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "active_task": None,
    "pid": None,
    "finished_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
printf '%s status=%s exit=%s\n' "$(date -Is)" "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_PATH}"
exit "${STATUS}"
