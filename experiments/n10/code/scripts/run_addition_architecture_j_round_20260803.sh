#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[01]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU SHARD_0_OR_1" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
SHARD="$2"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/architecture_round_20260803
ELIGIBILITY_JSON="${RUN_ROOT}/balanced_carry_screen/aggregate/aggregate.json"
J_ROOT="${RUN_ROOT}/carry_j_round"
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
MANIFEST_PATH="${J_ROOT}/shard${SHARD}_manifest.json"
HEARTBEAT_PATH="${J_ROOT}/shard${SHARD}_heartbeat.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

if [[ ! -f "${ELIGIBILITY_JSON}" ]]; then
    echo "missing eligibility aggregate: ${ELIGIBILITY_JSON}" >&2
    exit 3
fi
mapfile -t ELIGIBLE_JOBS < <("${PYTHON_BIN}" - "${ELIGIBILITY_JSON}" <<'PY'
import json, sys
for job in json.load(open(sys.argv[1]))["eligible_jobs"]:
    print(job)
PY
)

mkdir -p "${J_ROOT}" "${LOG_ROOT}"
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

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${SHARD}" \
    "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" "${ACTIVE_PIDS}" \
    "${ELIGIBLE_JOBS[@]}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "running", "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]), "shard": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]), "prelaunch_free_mib": int(sys.argv[5]),
    "prelaunch_utilization_percent": int(sys.argv[6]),
    "preexisting_pids": sys.argv[7].split(), "eligible_jobs": sys.argv[8:],
    "declared_peak_mib": 6144, "reserve_mib": 16384,
    "controller": "J(h)=hD+(hA)B+b, rank 48, shared before loops 2 onward",
    "initialization": "identity", "anchor_step": 1, "post_final_j": False,
    "supervision": "balanced final-carry CE at registered T(k)=k+1, k=1..10",
    "optimizer_updates": 5376,
    "wsd": {"warmup": 2048, "stable": 2816, "decay": 512},
    "peak_learning_rate": 1e-4, "diagonal_learning_rate": 1e-5,
    "completed_jobs": [],
}, indent=2, sort_keys=True) + "\n")
PY

update_manifest() {
    local status="$1" job="$2" phase="$3" pid="$4" log_path="$5"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${job}" "${phase}" "${pid}" "${log_path}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1]); data = json.loads(path.read_text())
data.update({"status": sys.argv[2], "active_job": sys.argv[3] or None,
             "active_phase": sys.argv[4] or None,
             "pid": int(sys.argv[5]) if sys.argv[5] else None,
             "active_log": sys.argv[6] or None, "updated_unix": time.time()})
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

complete_job() {
    local job="$1"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${job}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1]); data = json.loads(path.read_text())
done = list(data.get("completed_jobs", []))
if sys.argv[2] not in done: done.append(sys.argv[2])
data.update({"completed_jobs": done, "updated_unix": time.time()})
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_monitored() {
    local job="$1" phase="$2" log_path="$3"
    shift 3
    local free_mib used_mib utilization child_pid status
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then return 75; fi
    printf 'launch_command=' >> "${log_path}"; printf '%q ' "$@" >> "${log_path}"; printf '\n' >> "${log_path}"
    "$@" >> "${log_path}" 2>&1 &
    child_pid=$!; update_manifest running "${job}" "${phase}" "${child_pid}" "${log_path}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        used_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        utilization="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${job}" "${phase}" "${child_pid}" \
            "${PHYSICAL_GPU}" "${used_mib}" "${free_mib}" "${utilization}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(), "active_job": sys.argv[2], "active_phase": sys.argv[3],
    "child_pid": int(sys.argv[4]), "physical_gpu": int(sys.argv[5]),
    "used_mib": int(sys.argv[6]), "free_mib": int(sys.argv[7]),
    "utilization_percent": int(sys.argv[8]),
}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        sleep 30
    done
    status=0; wait "${child_pid}" || status=$?
    [[ "${status}" -eq 0 ]] || { update_manifest failed "${job}" "${phase}" "" "${log_path}"; return "${status}"; }
}

for JOB_INDEX in "${!ELIGIBLE_JOBS[@]}"; do
    if (( JOB_INDEX % 2 != SHARD )); then continue; fi
    JOB="${ELIGIBLE_JOBS[JOB_INDEX]}"
    SEED="${JOB##*_seed}"
    if [[ "${JOB}" == l3h8t11_step40k_seed* ]]; then
        BACKBONE="/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed${SEED}/checkpoint_040000.pt"
    elif [[ "${JOB}" == l3h8t11_seed* ]]; then
        BACKBONE="/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed${SEED}/checkpoint_010000.pt"
    else
        BACKBONE="${RUN_ROOT}/backbones/${JOB}/final.pt"
    fi
    if [[ ! -f "${BACKBONE}" ]]; then echo "missing ${BACKBONE}" >&2; exit 4; fi
    OUT_DIR="${J_ROOT}/${JOB}"
    CONTROLLER="${OUT_DIR}/controller.pt"
    CONTROLLER_SEED=$((421001 + SEED))
    LOG_PATH="${LOG_ROOT}/addition_architecture_carry_j_${JOB}.log"
    mkdir -p "${OUT_DIR}"

    if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" train_j_5376 "${LOG_PATH}" \
            "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
            --checkpoint "${BACKBONE}" --rank 48 --seed "${CONTROLLER_SEED}" \
            --controller-curriculum logical_range \
            --controller-logical-min-length 1 --controller-logical-max-length 10 \
            --controller-training-profile identity_long_warmup \
            --controller-initialization identity --controller-anchor-step 1 \
            --controller-warmup-updates 2048 --controller-final-lr-ratio 0.1 \
            --controller-lr-schedule wsd --controller-stable-updates 2816 \
            --dense-stage-count 0 --grad-clip 1.0 \
            --learning-rate-multiplier 5.0 --diagonal-lr-multiplier 0.1 \
            --stage-round-multiplier 3 \
            --controller-supervision addition_final_carry \
            --force --device cuda --out-dir "${OUT_DIR}"
    fi
    [[ -f "${CONTROLLER}" ]] || { echo "missing controller ${CONTROLLER}" >&2; exit 5; }

    DENSE_DIR="${OUT_DIR}/balanced_carry_dense_l1to50"
    if [[ ! -f "${DENSE_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" balanced_dense_l1to50 "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
            --addition-final-carry --lengths {1..50} --examples 128 \
            --max-batch-size 32 --token-budget 8192 --seed 511001 --device cuda \
            --out-dir "${DENSE_DIR}"
    fi

    ANCHOR_DIR="${OUT_DIR}/balanced_carry_anchor512"
    if [[ ! -f "${ANCHOR_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" balanced_anchor512 "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
            --addition-final-carry --lengths 1 5 10 11 12 15 20 25 30 40 50 75 100 \
            --examples 512 --max-batch-size 32 --token-budget 8192 \
            --seed 512001 --device cuda --out-dir "${ANCHOR_DIR}"
    fi

    FULL_DIR="${OUT_DIR}/full_answer_audit256"
    if [[ ! -f "${FULL_DIR}/summary.json" ]]; then
        run_monitored "${JOB}" full_answer_audit256 "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
            --lengths 10 15 20 30 40 --examples 256 --max-batch-size 32 \
            --token-budget 8192 --seed 513001 --device cuda --out-dir "${FULL_DIR}"
    fi

    for CONTROL in full no_AB identity_D no_bias executor_off; do
        CONTROL_DIR="${OUT_DIR}/control_${CONTROL}_256"
        if [[ -f "${CONTROL_DIR}/summary.json" ]]; then continue; fi
        CONTROL_ARGS=()
        if [[ "${CONTROL}" == full ]]; then
            CONTROL_ARGS=()
        elif [[ "${CONTROL}" == executor_off ]]; then
            CONTROL_ARGS=(--executor-off-after-anchor)
        else
            CONTROL_ARGS=(--controller-mode "${CONTROL}")
        fi
        run_monitored "${JOB}" "control_${CONTROL}_256" "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" --controller "${CONTROLLER}" \
            --addition-final-carry --lengths 10 15 20 30 40 75 \
            --examples 256 --max-batch-size 32 --token-budget 8192 \
            --seed 514001 --device cuda --out-dir "${CONTROL_DIR}" \
            "${CONTROL_ARGS[@]}"
    done
    complete_job "${JOB}"
done

update_manifest complete "" "" "" ""
