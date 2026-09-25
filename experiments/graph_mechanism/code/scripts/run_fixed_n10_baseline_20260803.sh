#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]] || \
   { [[ "$1" != "addition" ]] && [[ "$1" != "sum_reverse" ]]; } || \
   ! [[ "$2" =~ ^[0-7]$ ]] || ! [[ "$3" =~ ^[0-2]$ ]]; then
    echo "usage: $0 {addition|sum_reverse} PHYSICAL_GPU SEED" >&2
    exit 2
fi

TASK="$1"
PHYSICAL_GPU="$2"
SEED="$3"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
TARGET_LOOPS=10
if [[ "${TASK}" == "addition" ]]; then TARGET_LOOPS=11; fi
LABEL="${TASK}_fixed_n10_t${TARGET_LOOPS}_official_seed${SEED}"
BACKBONE_DIR="${RUN_ROOT}/backbones/${LABEL}"
BACKBONE="${BACKBONE_DIR}/final.pt"
MANIFEST_PATH="${BACKBONE_DIR}/pipeline_manifest.json"
HEARTBEAT_PATH="${BACKBONE_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${BACKBONE_DIR}" "${LOG_ROOT}"
PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "GPU ${PHYSICAL_GPU} has ${PRELAUNCH_FREE_MIB}MiB free; ${REQUIRED_FREE_MIB}MiB required" >&2
    exit 75
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.075
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${TASK}" "${SEED}" \
    "${PHYSICAL_GPU}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" "${BACKBONE_DIR}" \
    "${TARGET_LOOPS}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "task": sys.argv[2],
    "seed": int(sys.argv[3]),
    "physical_gpu": int(sys.argv[4]),
    "prelaunch_used_mib": int(sys.argv[5]),
    "prelaunch_free_mib": int(sys.argv[6]),
    "prelaunch_utilization_percent": int(sys.argv[7]),
    "preexisting_pids": sys.argv[8].split(),
    "backbone_dir": sys.argv[9],
    "declared_matching_peak_mib": 6144,
    "reserve_mib": 16384,
    "official_model_config": True,
    "training_precision": "fp32",
    "supervision": "final answer-region CE only",
    "training_logical_length_support": [10, 10],
    "training_target_loops": int(sys.argv[10]),
    "total_optimizer_updates": 100001,
    "batch_size": 64,
    "total_training_examples": 6400064,
    "completed_phases": [],
}, indent=2, sort_keys=True) + "\n")
PY

pipeline_exit() {
    local status="$1"
    if [[ "${status}" -eq 0 ]] || [[ ! -f "${MANIFEST_PATH}" ]]; then return; fi
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" <<'PY' || true
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "finished_unix": time.time(),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}
trap 'pipeline_exit $?' EXIT

manifest_phase() {
    local status="$1" phase="$2" child_pid="$3" log_path="$4"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${phase}" "${child_pid}" "${log_path}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "active_phase": sys.argv[3],
    "pid": int(sys.argv[4]),
    "active_log": sys.argv[5],
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

complete_phase() {
    local phase="$1"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${phase}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
completed = list(payload.get("completed_phases", []))
if sys.argv[2] not in completed:
    completed.append(sys.argv[2])
payload.update({"completed_phases": completed, "updated_unix": time.time()})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_monitored() {
    local phase="$1" log_path="$2"
    shift 2
    local free_mib child_pid status used_mib utilization
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "GPU ${PHYSICAL_GPU} has ${free_mib}MiB free before ${phase}; ${REQUIRED_FREE_MIB}MiB required" >&2
        return 75
    fi
    printf 'launch_command=' >> "${log_path}"
    printf '%q ' "$@" >> "${log_path}"
    printf '\n' >> "${log_path}"
    "$@" >> "${log_path}" 2>&1 &
    child_pid=$!
    manifest_phase running "${phase}" "${child_pid}" "${log_path}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        used_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        utilization="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${phase}" "${child_pid}" \
            "${PHYSICAL_GPU}" "${used_mib}" "${free_mib}" "${utilization}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(),
    "phase": sys.argv[2],
    "child_pid": int(sys.argv[3]),
    "physical_gpu": int(sys.argv[4]),
    "used_mib": int(sys.argv[5]),
    "free_mib": int(sys.argv[6]),
    "utilization_percent": int(sys.argv[7]),
}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            echo "reserve guard triggered in ${phase}: free=${free_mib}MiB" >> "${log_path}"
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        sleep 30
    done
    status=0
    wait "${child_pid}" || status=$?
    if [[ "${status}" -eq 0 ]]; then complete_phase "${phase}"; fi
    return "${status}"
}

if [[ ! -f "${BACKBONE_DIR}/summary.json" ]] || \
   ! grep -q '"status": "complete"' "${BACKBONE_DIR}/summary.json"; then
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${BACKBONE_DIR}" -maxdepth 1 -name 'checkpoint_*.pt' -type f | sort | tail -n 1)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then RESUME_ARGS=(--resume "${LATEST_CHECKPOINT}"); fi
    run_monitored train_100001 "${LOG_ROOT}/${LABEL}.log" \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone \
        --task "${TASK}" \
        --supervision adaptive_step \
        --steps 100001 \
        --batch-size 64 \
        --official-model-config \
        --train-fixed-logical-length 10 \
        --no-amp \
        --device cuda \
        --seed "${SEED}" \
        --log-every 100 \
        --eval-every 1000 \
        --checkpoint-every 10000 \
        --out-dir "${BACKBONE_DIR}" \
        "${RESUME_ARGS[@]}"
else
    complete_phase train_100001
fi

if [[ ! -f "${BACKBONE}" ]]; then
    echo "backbone was not produced" >&2
    exit 3
fi

DENSE_DIR="${RUN_ROOT}/dense_horizon_20260803/${LABEL}_dense64_l1to100"
if [[ ! -f "${DENSE_DIR}/summary.json" ]]; then
    DENSE_LENGTHS=()
    for ((length = 1; length <= 100; length += 1)); do DENSE_LENGTHS+=("${length}"); done
    run_monitored dense_eval_1to100 "${LOG_ROOT}/${LABEL}_dense64_l1to100.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" \
        --checkpoint "${BACKBONE}" \
        --lengths "${DENSE_LENGTHS[@]}" \
        --examples 64 \
        --max-batch-size 32 \
        --token-budget 8192 \
        --seed 271001 \
        --device cuda \
        --out-dir "${DENSE_DIR}"
else
    complete_phase dense_eval_1to100
fi

ANCHOR_DIR="${RUN_ROOT}/dense_horizon_20260803/${LABEL}_anchor512"
if [[ ! -f "${ANCHOR_DIR}/summary.json" ]]; then
    run_monitored anchor_eval_512 "${LOG_ROOT}/${LABEL}_anchor512.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" \
        --checkpoint "${BACKBONE}" \
        --lengths 1 5 10 15 20 25 30 35 40 50 60 \
        --examples 512 \
        --max-batch-size 32 \
        --token-budget 8192 \
        --seed 381001 \
        --device cuda \
        --out-dir "${ANCHOR_DIR}"
else
    complete_phase anchor_eval_512
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json, pathlib, time, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "active_phase": None,
    "pid": None,
    "finished_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
