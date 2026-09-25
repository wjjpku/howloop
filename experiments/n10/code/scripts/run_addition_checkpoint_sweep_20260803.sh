#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[0-2]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU BACKBONE_SEED" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
BACKBONE_SEED="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/checkpoint_sweep_20260803
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
SEED_ROOT="${RUN_ROOT}/seed${BACKBONE_SEED}"
MANIFEST_PATH="${SEED_ROOT}/pipeline_manifest.json"
HEARTBEAT_PATH="${SEED_ROOT}/heartbeat.json"
LOG_PATH="${LOG_ROOT}/addition_checkpoint_sweep_seed${BACKBONE_SEED}.log"
DECLARED_PEAK_MIB=256
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
CHECKPOINT_STEPS=(010000 020000 030000 040000 050000 100000)

mkdir -p "${SEED_ROOT}" "${LOG_ROOT}"
FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ "${FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "GPU ${PHYSICAL_GPU}: free=${FREE_MIB}MiB, required=${REQUIRED_FREE_MIB}MiB" >&2
    exit 75
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.075
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${BACKBONE_SEED}" "${PHYSICAL_GPU}" \
    "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "backbone_seed": int(sys.argv[2]),
    "physical_gpu": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]),
    "prelaunch_free_mib": int(sys.argv[5]),
    "prelaunch_utilization_percent": int(sys.argv[6]),
    "preexisting_pids": sys.argv[7].split(),
    "declared_measured_peak_mib": 256,
    "reserve_mib": 16384,
    "checkpoint_steps": [10000, 20000, 30000, 40000, 50000, 100000],
    "completed_checkpoint_steps": [],
    "evaluation_lengths": list(range(10, 41)),
    "examples_per_length": 512,
    "evaluation_seed": 291001,
    "step_offsets": [-2, -1, 0, 1, 2],
    "full_trajectory_lengths": [10, 15, 20],
}, indent=2, sort_keys=True) + "\n")
PY

update_manifest() {
    local status="$1" active_step="$2" child_pid="$3"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${active_step}" "${child_pid}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "active_checkpoint_step": int(sys.argv[3]) if sys.argv[3] else None,
    "pid": int(sys.argv[4]) if sys.argv[4] else None,
    "active_log": str(pathlib.Path("/data/wujiaju/logs/paper_length_telomere_20260731") / f"addition_checkpoint_sweep_seed{payload['backbone_seed']}.log"),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

complete_step() {
    local completed_step="$1"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${completed_step}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
completed = list(payload.get("completed_checkpoint_steps", []))
step = int(sys.argv[2])
if step not in completed:
    completed.append(step)
payload.update({"completed_checkpoint_steps": sorted(completed), "updated_unix": time.time()})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

for PADDED_STEP in "${CHECKPOINT_STEPS[@]}"; do
    STEP=$((10#${PADDED_STEP}))
    CHECKPOINT="/data/wujiaju/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed${BACKBONE_SEED}/checkpoint_${PADDED_STEP}.pt"
    OUT_DIR="${SEED_ROOT}/step_${PADDED_STEP}"
    if [[ -f "${OUT_DIR}/summary.json" ]] && grep -q '"status": "complete"' "${OUT_DIR}/summary.json"; then
        complete_step "${STEP}"
        continue
    fi
    mkdir -p "${OUT_DIR}"
    printf 'checkpoint_step=%s launch_command=' "${STEP}" >> "${LOG_PATH}"
    printf '%q ' "${PYTHON_BIN}" -m scripts.evaluate_addition_checkpoint_sweep \
        --checkpoint "${CHECKPOINT}" --lengths {10..40} --examples 512 \
        --max-batch-size 64 --token-budget 8192 --step-offsets -2 -1 0 1 2 \
        --full-trajectory-lengths 10 15 20 --seed 291001 --device cuda \
        --out-dir "${OUT_DIR}" >> "${LOG_PATH}"
    printf '\n' >> "${LOG_PATH}"
    "${PYTHON_BIN}" -m scripts.evaluate_addition_checkpoint_sweep \
        --checkpoint "${CHECKPOINT}" \
        --lengths {10..40} \
        --examples 512 \
        --max-batch-size 64 \
        --token-budget 8192 \
        --step-offsets -2 -1 0 1 2 \
        --full-trajectory-lengths 10 15 20 \
        --seed 291001 \
        --device cuda \
        --out-dir "${OUT_DIR}" >> "${LOG_PATH}" 2>&1 &
    CHILD_PID=$!
    update_manifest running "${STEP}" "${CHILD_PID}"
    while kill -0 "${CHILD_PID}" 2>/dev/null; do
        USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${STEP}" "${CHILD_PID}" \
            "${PHYSICAL_GPU}" "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(),
    "active_checkpoint_step": int(sys.argv[2]),
    "child_pid": int(sys.argv[3]),
    "physical_gpu": int(sys.argv[4]),
    "used_mib": int(sys.argv[5]),
    "free_mib": int(sys.argv[6]),
    "utilization_percent": int(sys.argv[7]),
}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${FREE_MIB}" -lt "${RESERVE_MIB}" ]]; then
            echo "reserve guard: free=${FREE_MIB}MiB at step=${STEP}" >> "${LOG_PATH}"
            kill -TERM "${CHILD_PID}" 2>/dev/null || true
            wait "${CHILD_PID}" || true
            update_manifest failed "${STEP}" ""
            exit 76
        fi
        sleep 30
    done
    STATUS=0
    wait "${CHILD_PID}" || STATUS=$?
    if [[ "${STATUS}" -ne 0 ]]; then
        update_manifest failed "${STEP}" ""
        exit "${STATUS}"
    fi
    complete_step "${STEP}"
done

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "active_checkpoint_step": None,
    "pid": None,
    "finished_unix": time.time(),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
