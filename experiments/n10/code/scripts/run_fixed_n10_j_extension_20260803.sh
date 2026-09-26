#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]] || \
   { [[ "$1" != "addition" ]] && [[ "$1" != "sum_reverse" ]]; } || \
   ! [[ "$2" =~ ^[0-2]$ ]] || ! [[ "$3" =~ ^[0-7]$ ]] || \
   ! [[ "$4" =~ ^[0-9]+$ ]] || ! [[ "$5" =~ ^[0-9]+$ ]] || \
   [[ "$4" -gt "$5" ]] || \
   { [[ "$6" != "full_answer" ]] && [[ "$6" != "addition_final_carry" ]]; }; then
    echo "usage: $0 {addition|sum_reverse} BACKBONE_SEED PHYSICAL_GPU TRAIN_MIN TRAIN_MAX {full_answer|addition_final_carry}" >&2
    exit 2
fi

TASK="$1"
BACKBONE_SEED="$2"
PHYSICAL_GPU="$3"
TRAIN_MIN="$4"
TRAIN_MAX="$5"
CONTROLLER_SUPERVISION="$6"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
TARGET_LOOPS=10
if [[ "${TASK}" == "addition" ]]; then TARGET_LOOPS=11; fi
BACKBONE_LABEL="${TASK}_fixed_n10_t${TARGET_LOOPS}_official_seed${BACKBONE_SEED}"
BACKBONE="${RUN_ROOT}/backbones/${BACKBONE_LABEL}/final.pt"
CONTROLLER_SEED=$((421001 + BACKBONE_SEED))
SUPERVISION_LABEL="fullanswer"
if [[ "${CONTROLLER_SUPERVISION}" == "addition_final_carry" ]]; then SUPERVISION_LABEL="balanced_finalcarry"; fi
LABEL="${BACKBONE_LABEL}_rank48_identity_${SUPERVISION_LABEL}_logical${TRAIN_MIN}to${TRAIN_MAX}_anchor1_interloop_nopostfinal_wsd5376_seed${CONTROLLER_SEED}"
OUT_DIR="${RUN_ROOT}/controllers/${LABEL}"
CONTROLLER="${OUT_DIR}/controller.pt"
MANIFEST_PATH="${OUT_DIR}/pipeline_manifest.json"
HEARTBEAT_PATH="${OUT_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" ]]; then
    echo "missing backbone: ${BACKBONE}" >&2
    exit 2
fi

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

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${TASK}" "${BACKBONE_SEED}" \
    "${PHYSICAL_GPU}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" "${BACKBONE}" \
    "${TRAIN_MIN}" "${TRAIN_MAX}" "${CONTROLLER_SEED}" "${CONTROLLER_SUPERVISION}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "task": sys.argv[2],
    "backbone_seed": int(sys.argv[3]),
    "physical_gpu": int(sys.argv[4]),
    "prelaunch_used_mib": int(sys.argv[5]),
    "prelaunch_free_mib": int(sys.argv[6]),
    "prelaunch_utilization_percent": int(sys.argv[7]),
    "preexisting_pids": sys.argv[8].split(),
    "backbone": sys.argv[9],
    "controller_seed": int(sys.argv[12]),
    "controller_supervision": sys.argv[13],
    "controller": "J(h)=hD+(hA)B+b, rank 48",
    "initialization": "identity",
    "anchor_step": 1,
    "application": "leave loop 1 raw, then apply the same J immediately before loops 2 onward",
    "post_final_j": False,
    "loss": (
        "balanced CE on only the carry-chain terminal bit at T(k)=k+1"
        if sys.argv[13] == "addition_final_carry"
        else "final registered T(n) full-answer-region CE"
    ),
    "addition_parallel_pairs": (
        [[length, length + 1] for length in range(int(sys.argv[10]), int(sys.argv[11]) + 1)]
        if sys.argv[2] == "addition" else None
    ),
    "controller_training_logical_range": [int(sys.argv[10]), int(sys.argv[11])],
    "strict_unseen_logical_range": [int(sys.argv[11]) + 1, 100],
    "optimizer_updates": 5376,
    "wsd": {"warmup": 2048, "stable": 2816, "decay": 512},
    "peak_learning_rate": 1e-4,
    "diagonal_learning_rate": 1e-5,
    "declared_matching_peak_mib": 6144,
    "reserve_mib": 16384,
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
payload.update({"status": "failed", "exit_code": int(sys.argv[2]), "finished_unix": time.time()})
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
payload.update({"status": sys.argv[2], "active_phase": sys.argv[3], "pid": int(sys.argv[4]), "active_log": sys.argv[5], "updated_unix": time.time()})
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
if sys.argv[2] not in completed: completed.append(sys.argv[2])
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
pathlib.Path(sys.argv[1]).write_text(json.dumps({"unix": time.time(), "phase": sys.argv[2], "child_pid": int(sys.argv[3]), "physical_gpu": int(sys.argv[4]), "used_mib": int(sys.argv[5]), "free_mib": int(sys.argv[6]), "utilization_percent": int(sys.argv[7])}, indent=2, sort_keys=True) + "\n")
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

if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
    SUPERVISION_ARGS=(--controller-supervision "${CONTROLLER_SUPERVISION}")
    run_monitored train_j_5376 "${LOG_ROOT}/${LABEL}.log" \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
        --checkpoint "${BACKBONE}" \
        --rank 48 \
        --seed "${CONTROLLER_SEED}" \
        --controller-curriculum logical_range \
        --controller-logical-min-length "${TRAIN_MIN}" \
        --controller-logical-max-length "${TRAIN_MAX}" \
        --controller-training-profile identity_long_warmup \
        --controller-initialization identity \
        --controller-anchor-step 1 \
        --controller-warmup-updates 2048 \
        --controller-final-lr-ratio 0.1 \
        --controller-lr-schedule wsd \
        --controller-stable-updates 2816 \
        --dense-stage-count 0 \
        --grad-clip 1.0 \
        --learning-rate-multiplier 5.0 \
        --diagonal-lr-multiplier 0.1 \
        --stage-round-multiplier 3 \
        "${SUPERVISION_ARGS[@]}" \
        --force \
        --device cuda \
        --out-dir "${OUT_DIR}"
else
    complete_phase train_j_5376
fi

if [[ ! -f "${CONTROLLER}" ]]; then
    echo "controller was not produced" >&2
    exit 3
fi

EVAL_SUPERVISION_ARGS=()
if [[ "${CONTROLLER_SUPERVISION}" == "addition_final_carry" ]]; then
    EVAL_SUPERVISION_ARGS=(--addition-final-carry)
fi

DENSE_DIR="${RUN_ROOT}/dense_horizon_20260803/${LABEL}_dense64_l1to100"
if [[ ! -f "${DENSE_DIR}/summary.json" ]]; then
    DENSE_LENGTHS=()
    for ((length = 1; length <= 100; length += 1)); do DENSE_LENGTHS+=("${length}"); done
    run_monitored dense_eval_1to100 "${LOG_ROOT}/${LABEL}_dense64_l1to100.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
        --lengths "${DENSE_LENGTHS[@]}" --examples 64 --max-batch-size 32 \
        --token-budget 8192 --seed 471001 --device cuda --out-dir "${DENSE_DIR}" \
        "${EVAL_SUPERVISION_ARGS[@]}"
else
    complete_phase dense_eval_1to100
fi

ANCHOR_DIR="${RUN_ROOT}/dense_horizon_20260803/${LABEL}_anchor512"
if [[ ! -f "${ANCHOR_DIR}/summary.json" ]]; then
    run_monitored anchor_eval_512 "${LOG_ROOT}/${LABEL}_anchor512.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task "${TASK}" --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
        --lengths 8 9 10 11 12 13 14 15 16 18 20 25 30 40 50 60 \
        --examples 512 --max-batch-size 32 --token-budget 8192 \
        --seed 481001 --device cuda --out-dir "${ANCHOR_DIR}" \
        "${EVAL_SUPERVISION_ARGS[@]}"
else
    complete_phase anchor_eval_512
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({"status": "complete", "active_phase": None, "pid": None, "finished_unix": time.time()})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
