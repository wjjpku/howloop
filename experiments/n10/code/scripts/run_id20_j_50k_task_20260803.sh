#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || \
   { [[ "$1" != "addition" ]] && [[ "$1" != "sum_reverse" ]]; } || \
   ! [[ "$2" =~ ^[0-7]$ ]]; then
    echo "usage: $0 {addition|sum_reverse} PHYSICAL_GPU" >&2
    exit 2
fi

TASK="$1"
PHYSICAL_GPU="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
CONTROLLER_SEED=211001
CONTINUATION_SEED=311001
INITIAL_LABEL="${TASK}_adaptive_step_official_seed0_rank48_identitywarmup_logical1to20_fullrange_anchor1_postfinal_wsd_lr1em4_warmup2048_stable2816_5k_seed${CONTROLLER_SEED}"
FINAL_LABEL="${TASK}_adaptive_step_official_seed0_rank48_logical1to20_fullrange_anchor1_postfinal_continue5376to50000_mix1_10_15_20_wsd5k_35k_4624_lr1em4_seed${CONTINUATION_SEED}"
BACKBONE="${RUN_ROOT}/backbones/${TASK}_adaptive_step_official_seed0/final.pt"
INITIAL_DIR="${RUN_ROOT}/controllers/${INITIAL_LABEL}"
INITIAL_CONTROLLER="${INITIAL_DIR}/controller.pt"
FINAL_DIR="${RUN_ROOT}/controllers/${FINAL_LABEL}"
FINAL_CONTROLLER="${FINAL_DIR}/controller.pt"
MANIFEST_PATH="${FINAL_DIR}/pipeline_manifest.json"
HEARTBEAT_PATH="${FINAL_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384

mkdir -p "${INITIAL_DIR}" "${FINAL_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" ]]; then
    echo "missing backbone: ${BACKBONE}" >&2
    exit 2
fi

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
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

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${TASK}" "${PHYSICAL_GPU}" \
    "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" "${BACKBONE}" \
    "${INITIAL_DIR}" "${FINAL_DIR}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "task": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]),
    "prelaunch_free_mib": int(sys.argv[5]),
    "prelaunch_utilization_percent": int(sys.argv[6]),
    "preexisting_pids": sys.argv[7].split(),
    "backbone": sys.argv[8],
    "initial_controller_dir": sys.argv[9],
    "final_controller_dir": sys.argv[10],
    "declared_matching_peak_mib": 6144,
    "reserve_mib": 16384,
    "loss": "final registered T(n) answer-region CE only",
    "controller": "J(h)=hD+(hA)B+b, rank 48",
    "initialization": "identity",
    "anchor_step": 1,
    "post_final_j": True,
    "strict_training_logical_range": [1, 20],
    "initial_updates": 5376,
    "target_total_updates": 50000,
    "total_training_examples": 1649152,
    "initial_wsd": {"warmup": 2048, "stable": 2816, "decay": 512},
    "continuation_wsd": {"warmup": 5000, "stable": 35000, "decay": 4624},
    "continuation_length_mix": {
        "1-10": 0.20,
        "11-15": 0.30,
        "16-20": 0.50,
    },
    "peak_learning_rate": 1e-4,
    "completed_phases": [],
}, indent=2, sort_keys=True) + "\n")
PY

pipeline_exit() {
    local status="$1"
    if [[ "${status}" -eq 0 ]] || [[ ! -f "${MANIFEST_PATH}" ]]; then
        return
    fi
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
    local status="$1"
    local phase="$2"
    local child_pid="$3"
    local log_path="$4"
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
    local phase="$1"
    local log_path="$2"
    shift 2
    local free_mib
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "GPU ${PHYSICAL_GPU} has ${free_mib}MiB free before ${phase}; ${REQUIRED_FREE_MIB}MiB required" >&2
        return 75
    fi
    printf 'launch_command=' >> "${log_path}"
    printf '%q ' "$@" >> "${log_path}"
    printf '\n' >> "${log_path}"
    "$@" >> "${log_path}" 2>&1 &
    local child_pid=$!
    manifest_phase running "${phase}" "${child_pid}" "${log_path}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        local used_mib utilization
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
    local status=0
    wait "${child_pid}" || status=$?
    if [[ "${status}" -eq 0 ]]; then
        complete_phase "${phase}"
    fi
    return "${status}"
}

if [[ ! -f "${INITIAL_DIR}/summary.json" ]] || \
   ! grep -q '"status": "complete"' "${INITIAL_DIR}/summary.json"; then
    run_monitored initial_5376 "${LOG_ROOT}/${INITIAL_LABEL}.log" \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
        --checkpoint "${BACKBONE}" \
        --rank 48 \
        --seed "${CONTROLLER_SEED}" \
        --controller-curriculum logical_range \
        --controller-logical-min-length 1 \
        --controller-logical-max-length 20 \
        --controller-training-profile identity_long_warmup \
        --controller-initialization identity \
        --controller-anchor-step 1 \
        --controller-warmup-updates 2048 \
        --controller-final-lr-ratio 0.1 \
        --controller-lr-schedule wsd \
        --controller-stable-updates 2816 \
        --controller-post-final-j \
        --dense-stage-count 0 \
        --grad-clip 1.0 \
        --learning-rate-multiplier 5.0 \
        --diagonal-lr-multiplier 0.1 \
        --stage-round-multiplier 3 \
        --force \
        --device cuda \
        --out-dir "${INITIAL_DIR}"
else
    complete_phase initial_5376
fi

if [[ ! -f "${INITIAL_CONTROLLER}" ]]; then
    echo "initial controller was not produced" >&2
    exit 3
fi

if [[ ! -f "${FINAL_DIR}/summary.json" ]] || \
   ! grep -q '"status": "complete"' "${FINAL_DIR}/summary.json"; then
    RESUME_ARGS=()
    if [[ -f "${FINAL_DIR}/latest.pt" ]]; then
        RESUME_ARGS=(--resume "${FINAL_DIR}/latest.pt")
    fi
    run_monitored continue_to_50000 "${LOG_ROOT}/${FINAL_LABEL}.log" \
        "${PYTHON_BIN}" -m reasoning_loop.continue_paper_length_controller \
        --task "${TASK}" \
        --checkpoint "${BACKBONE}" \
        --source-controller "${INITIAL_CONTROLLER}" \
        --start-total-update 5376 \
        --target-total-update 50000 \
        --batch-size 32 \
        --retention-maximum 10 \
        --transition-maximum 15 \
        --boundary-maximum 20 \
        --retention-probability 0.20 \
        --transition-probability 0.30 \
        --boundary-probability 0.50 \
        --peak-learning-rate 1e-4 \
        --diagonal-lr-multiplier 0.1 \
        --warmup-updates 5000 \
        --stable-updates 35000 \
        --final-learning-rate-ratio 0.1 \
        --grad-clip 1.0 \
        --seed "${CONTINUATION_SEED}" \
        --eval-seed 361001 \
        --eval-lengths 1 5 10 15 19 20 25 30 35 40 50 60 \
        --eval-batch-size 128 \
        --eval-batches 2 \
        --eval-every 1000 \
        --log-every 100 \
        --checkpoint-every 5000 \
        --device cuda \
        --out-dir "${FINAL_DIR}" \
        "${RESUME_ARGS[@]}"
else
    complete_phase continue_to_50000
fi

if [[ ! -f "${FINAL_CONTROLLER}" ]]; then
    echo "final controller was not produced" >&2
    exit 3
fi

DENSE_DIR="${RUN_ROOT}/dense_horizon_20260803/${TASK}_seed0_id20_50k_dense64_l1to100"
if [[ ! -f "${DENSE_DIR}/summary.json" ]]; then
    DENSE_LENGTHS=()
    for ((length = 1; length <= 100; length += 1)); do DENSE_LENGTHS+=("${length}"); done
    run_monitored dense_eval_1to100 "${LOG_ROOT}/${TASK}_seed0_id20_50k_dense64_l1to100.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" \
        --checkpoint "${BACKBONE}" \
        --controller "${FINAL_CONTROLLER}" \
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

ANCHOR_DIR="${RUN_ROOT}/dense_horizon_20260803/${TASK}_seed0_id20_50k_anchor512"
if [[ ! -f "${ANCHOR_DIR}/summary.json" ]]; then
    run_monitored anchor_eval_512 "${LOG_ROOT}/${TASK}_seed0_id20_50k_anchor512.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" \
        --checkpoint "${BACKBONE}" \
        --controller "${FINAL_CONTROLLER}" \
        --lengths 1 5 10 15 19 20 25 30 35 40 45 50 60 \
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
