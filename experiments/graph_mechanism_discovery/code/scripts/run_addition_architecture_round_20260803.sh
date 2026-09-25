#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[01]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU SHARD_0_OR_1 [JOB ...]" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
SHARD="$2"
JOB_FILTER=("${@:3}")
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/architecture_round_20260803
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
RUNNER_INSTANCE="${ARCH_RUNNER_INSTANCE:-shard${SHARD}}"
SHARD_ROOT="${RUN_ROOT}/${RUNNER_INSTANCE}"
MANIFEST_PATH="${SHARD_ROOT}/pipeline_manifest.json"
HEARTBEAT_PATH="${SHARD_ROOT}/heartbeat.json"
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

CONFIGS=(l2h8t11 l3h4t11 l2h4t11 l3h8t8 l3h8t6 l2h4t8)
ALL_JOBS=()
for CONFIG_NAME in "${CONFIGS[@]}"; do
    for BACKBONE_SEED in 0 1 2; do
        ALL_JOBS+=("${CONFIG_NAME}_seed${BACKBONE_SEED}")
    done
done

mkdir -p "${SHARD_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/backbones" "${RUN_ROOT}/eval"
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

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${SHARD}" "${PHYSICAL_GPU}" \
    "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "shard": int(sys.argv[2]),
    "physical_gpu": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]),
    "prelaunch_free_mib": int(sys.argv[5]),
    "prelaunch_utilization_percent": int(sys.argv[6]),
    "preexisting_pids": sys.argv[7].split(),
    "declared_matching_peak_mib": 2048,
    "reserve_mib": 16384,
    "backbone_steps": 10000,
    "schedule_total_steps": 100001,
    "batch_size": 64,
    "training_logical_length": 10,
    "training_precision": "fp32",
    "completed_jobs": [],
}, indent=2, sort_keys=True) + "\n")
PY

update_manifest() {
    local status="$1" active_job="$2" phase="$3" child_pid="$4" log_path="$5"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${active_job}" \
        "${phase}" "${child_pid}" "${log_path}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "active_job": sys.argv[3] or None,
    "active_phase": sys.argv[4] or None,
    "pid": int(sys.argv[5]) if sys.argv[5] else None,
    "active_log": sys.argv[6] or None,
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

complete_job() {
    local job="$1"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${job}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
completed = list(payload.get("completed_jobs", []))
if sys.argv[2] not in completed:
    completed.append(sys.argv[2])
payload.update({"completed_jobs": completed, "updated_unix": time.time()})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_monitored() {
    local job="$1" phase="$2" log_path="$3"
    shift 3
    local child_pid free_mib used_mib utilization status
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "GPU ${PHYSICAL_GPU}: free=${free_mib}MiB before ${job}/${phase}" >&2
        return 75
    fi
    printf 'job=%s phase=%s launch_command=' "${job}" "${phase}" >> "${log_path}"
    printf '%q ' "$@" >> "${log_path}"
    printf '\n' >> "${log_path}"
    "$@" >> "${log_path}" 2>&1 &
    child_pid=$!
    update_manifest running "${job}" "${phase}" "${child_pid}" "${log_path}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        used_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        utilization="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${job}" "${phase}" \
            "${child_pid}" "${PHYSICAL_GPU}" "${used_mib}" "${free_mib}" \
            "${utilization}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(),
    "active_job": sys.argv[2],
    "active_phase": sys.argv[3],
    "child_pid": int(sys.argv[4]),
    "physical_gpu": int(sys.argv[5]),
    "used_mib": int(sys.argv[6]),
    "free_mib": int(sys.argv[7]),
    "utilization_percent": int(sys.argv[8]),
}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            echo "reserve guard: free=${free_mib}MiB at ${job}/${phase}" >> "${log_path}"
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            update_manifest failed "${job}" "${phase}" "" "${log_path}"
            return 76
        fi
        sleep 30
    done
    status=0
    wait "${child_pid}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        update_manifest failed "${job}" "${phase}" "" "${log_path}"
    fi
    return "${status}"
}

for JOB_INDEX in "${!ALL_JOBS[@]}"; do
    if (( JOB_INDEX % 2 != SHARD )); then
        continue
    fi
    JOB="${ALL_JOBS[JOB_INDEX]}"
    if [[ "${#JOB_FILTER[@]}" -gt 0 ]]; then
        MATCHED=0
        for FILTER_JOB in "${JOB_FILTER[@]}"; do
            if [[ "${JOB}" == "${FILTER_JOB}" ]]; then MATCHED=1; break; fi
        done
        if [[ "${MATCHED}" -eq 0 ]]; then continue; fi
    fi
    CONFIG_NAME="${JOB%_seed*}"
    BACKBONE_SEED="${JOB##*_seed}"
    case "${CONFIG_NAME}" in
        l2h8t11) BLOCK_LAYERS=2; HEADS=8; TRAIN_LOOPS=11 ;;
        l3h4t11) BLOCK_LAYERS=3; HEADS=4; TRAIN_LOOPS=11 ;;
        l2h4t11) BLOCK_LAYERS=2; HEADS=4; TRAIN_LOOPS=11 ;;
        l3h8t8)  BLOCK_LAYERS=3; HEADS=8; TRAIN_LOOPS=8 ;;
        l3h8t6)  BLOCK_LAYERS=3; HEADS=8; TRAIN_LOOPS=6 ;;
        l2h4t8)  BLOCK_LAYERS=2; HEADS=4; TRAIN_LOOPS=8 ;;
        *) echo "unknown config ${CONFIG_NAME}" >&2; exit 3 ;;
    esac
    BACKBONE_DIR="${RUN_ROOT}/backbones/${JOB}"
    BACKBONE="${BACKBONE_DIR}/final.pt"
    LOG_PATH="${LOG_ROOT}/addition_architecture_${JOB}.log"
    mkdir -p "${BACKBONE_DIR}"
    if [[ ! -f "${BACKBONE_DIR}/summary.json" ]] || \
       ! grep -q '"status": "complete"' "${BACKBONE_DIR}/summary.json"; then
        run_monitored "${JOB}" train_10k "${LOG_PATH}" \
            "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone \
            --task addition \
            --supervision fixed_horizon \
            --steps 10000 \
            --schedule-total-steps 100001 \
            --batch-size 64 \
            --learning-rate 1e-4 \
            --weight-decay 0.01 \
            --grad-clip 1.0 \
            --d-model 256 \
            --n-heads "${HEADS}" \
            --d-mlp 1024 \
            --block-layers "${BLOCK_LAYERS}" \
            --train-fixed-logical-length 10 \
            --train-fixed-loop-count "${TRAIN_LOOPS}" \
            --no-amp \
            --device cuda \
            --seed "${BACKBONE_SEED}" \
            --log-every 100 \
            --eval-every 1000 \
            --eval-batch-size 256 \
            --eval-batches 4 \
            --checkpoint-every 20000 \
            --out-dir "${BACKBONE_DIR}"
    fi
    if [[ ! -f "${BACKBONE}" ]]; then
        echo "missing backbone ${BACKBONE}" >&2
        exit 4
    fi

    EVAL_OFFSET=$((TRAIN_LOOPS - 10))
    ALIGNED_DIR="${RUN_ROOT}/eval/${JOB}_aligned"
    if [[ ! -f "${ALIGNED_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" eval_aligned "${LOG_PATH}" \
            "${PYTHON_BIN}" -m scripts.evaluate_addition_checkpoint_sweep \
            --checkpoint "${BACKBONE}" \
            --lengths {10..40} \
            --examples 512 \
            --max-batch-size 64 \
            --token-budget 8192 \
            --step-offsets -2 -1 0 1 2 \
            --evaluation-step-offset "${EVAL_OFFSET}" \
            --full-trajectory-lengths 10 15 20 \
            --seed 291001 \
            --device cuda \
            --out-dir "${ALIGNED_DIR}"
    fi

    REGISTERED_DIR="${RUN_ROOT}/eval/${JOB}_registered"
    if [[ ! -f "${REGISTERED_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" eval_registered "${LOG_PATH}" \
            "${PYTHON_BIN}" -m scripts.evaluate_addition_checkpoint_sweep \
            --checkpoint "${BACKBONE}" \
            --lengths {10..40} \
            --examples 512 \
            --max-batch-size 64 \
            --token-budget 8192 \
            --step-offsets -2 -1 0 1 2 \
            --full-trajectory-lengths 10 15 20 \
            --seed 291001 \
            --device cuda \
            --out-dir "${REGISTERED_DIR}"
    fi
    complete_job "${JOB}"
done

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "active_job": None,
    "active_phase": None,
    "pid": None,
    "finished_unix": time.time(),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
