#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[0-2]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU SHARD_0_TO_2" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
SHARD="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/architecture_round_20260803
J_ROOT="${RUN_ROOT}/fullanswer_j_round"
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731/fullanswer_architecture_j
MANIFEST_PATH="${J_ROOT}/shard${SHARD}_manifest.json"
HEARTBEAT_PATH="${J_ROOT}/shard${SHARD}_heartbeat.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
JOBS=(
    l2h8t11_seed0 l2h8t11_seed1 l2h8t11_seed2
    l3h4t11_seed0 l3h4t11_seed1 l3h4t11_seed2
    l2h4t11_seed0 l2h4t11_seed1 l2h4t11_seed2
    l3h8t8_seed0 l3h8t8_seed1 l3h8t8_seed2
    l3h8t6_seed0 l3h8t6_seed1 l3h8t6_seed2
    l2h4t8_seed0 l2h4t8_seed1 l2h4t8_seed2
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
    "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "running", "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]), "shard": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]),
    "prelaunch_free_mib": int(sys.argv[5]),
    "prelaunch_utilization_percent": int(sys.argv[6]),
    "preexisting_pids": sys.argv[7].split(),
    "declared_peak_mib": 6144, "reserve_mib": 16384,
    "controller": "J(h)=hD+(hA)B+b, rank 48, shared before loops 2 onward",
    "initialization": "exact identity", "anchor_step": 1,
    "post_final_j": False,
    "loss": "full answer-region CE at registered T(k)=k+1 for k=1..10",
    "optimizer_updates": 5376,
    "snapshots": "update 0, every 256 updates, and update 5376",
    "selection": "ID lengths 1..10 only; per-length EM gate 0.99 then minimum mean CE",
    "wsd": {"warmup": 2048, "stable": 2816, "decay": 512},
    "peak_learning_rate": 1e-4, "diagonal_learning_rate": 1e-5,
    "evaluated_logical_lengths": [1, 40],
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
if sys.argv[2] in {"complete", "failed"}: data["finished_unix"] = time.time()
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

pipeline_exit() {
    local status="$1"
    if [[ "${status}" -eq 0 ]]; then return; fi
    update_manifest failed "${ACTIVE_JOB:-}" "${ACTIVE_PHASE:-}" "" "${ACTIVE_LOG:-}" || true
}
trap 'pipeline_exit $?' EXIT

run_monitored() {
    local job="$1" phase="$2" log_path="$3"
    shift 3
    local free_mib used_mib utilization child_pid status
    ACTIVE_JOB="${job}"; ACTIVE_PHASE="${phase}"; ACTIVE_LOG="${log_path}"
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
        # Preserve the ~30 second heartbeat cadence for long phases without
        # imposing a 30 second bubble after every short snapshot evaluation.
        for _ in {1..15}; do
            sleep 2
            if ! kill -0 "${child_pid}" 2>/dev/null; then break; fi
        done
    done
    status=0; wait "${child_pid}" || status=$?
    [[ "${status}" -eq 0 ]] || return "${status}"
}

for JOB_INDEX in "${!JOBS[@]}"; do
    if (( JOB_INDEX % 3 != SHARD )); then continue; fi
    JOB="${JOBS[JOB_INDEX]}"
    SEED="${JOB##*_seed}"
    BACKBONE="${RUN_ROOT}/backbones/${JOB}/final.pt"
    OUT_DIR="${J_ROOT}/${JOB}"
    CONTROLLER="${OUT_DIR}/controller.pt"
    ID_ROOT="${OUT_DIR}/snapshot_id_eval_l1to10"
    SELECTION="${OUT_DIR}/snapshot_selection.json"
    LOG_PATH="${LOG_ROOT}/${JOB}.log"
    mkdir -p "${OUT_DIR}" "${ID_ROOT}"
    [[ -f "${BACKBONE}" ]] || { echo "missing ${BACKBONE}" >&2; exit 4; }

    if [[ ! -f "${CONTROLLER}" ]]; then
        run_monitored "${JOB}" train_fullanswer_j_5376 "${LOG_PATH}" \
            "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
            --checkpoint "${BACKBONE}" --rank 48 --seed "$((621001 + SEED))" \
            --controller-curriculum logical_range \
            --controller-logical-min-length 1 --controller-logical-max-length 10 \
            --controller-training-profile identity_long_warmup \
            --controller-initialization identity --controller-anchor-step 1 \
            --controller-warmup-updates 2048 --controller-final-lr-ratio 0.1 \
            --controller-lr-schedule wsd --controller-stable-updates 2816 \
            --controller-supervision full_answer --controller-checkpoint-every 256 \
            --dense-stage-count 0 --grad-clip 1.0 \
            --learning-rate-multiplier 5.0 --diagonal-lr-multiplier 0.1 \
            --stage-round-multiplier 3 --force --device cuda --out-dir "${OUT_DIR}"
    fi

    for SNAPSHOT in "${OUT_DIR}"/checkpoints/controller_*.pt; do
        SNAPSHOT_LABEL="$(basename "${SNAPSHOT}" .pt)"
        SNAPSHOT_DIR="${ID_ROOT}/${SNAPSHOT_LABEL}"
        if [[ -f "${SNAPSHOT_DIR}/summary.json" ]]; then continue; fi
        run_monitored "${JOB}" "id_${SNAPSHOT_LABEL}" "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" --controller "${SNAPSHOT}" \
            --lengths 1 2 3 4 5 6 7 8 9 10 --examples 128 \
            --max-batch-size 32 --token-budget 8192 --seed 631001 \
            --device cuda --out-dir "${SNAPSHOT_DIR}"
    done

    "${PYTHON_BIN}" scripts/select_controller_snapshot.py \
        --evaluation-root "${ID_ROOT}" --lengths 1 2 3 4 5 6 7 8 9 10 \
        --minimum-exact-match 0.99 --out "${SELECTION}" >> "${LOG_PATH}" 2>&1
    SELECTED_CONTROLLER="$("${PYTHON_BIN}" - "${SELECTION}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected"]["controller"])
PY
)"

    HORIZON_DIR="${OUT_DIR}/selected_fullanswer_l1to40_512"
    if [[ ! -f "${HORIZON_DIR}/summary.json" ]]; then
        LENGTHS=(); for ((n=1; n<=40; n+=1)); do LENGTHS+=("${n}"); done
        run_monitored "${JOB}" selected_fullanswer_l1to40_512 "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" \
            --controller "${SELECTED_CONTROLLER}" --lengths "${LENGTHS[@]}" \
            --examples 512 --max-batch-size 32 --token-budget 8192 \
            --seed 641001 --device cuda --out-dir "${HORIZON_DIR}"
    fi

    for CONTROL in no_AB identity_D no_bias executor_off; do
        CONTROL_DIR="${OUT_DIR}/control_${CONTROL}_256"
        if [[ -f "${CONTROL_DIR}/summary.json" ]]; then continue; fi
        CONTROL_ARGS=()
        if [[ "${CONTROL}" == executor_off ]]; then
            CONTROL_ARGS=(--executor-off-after-anchor)
        else
            CONTROL_ARGS=(--controller-mode "${CONTROL}")
        fi
        run_monitored "${JOB}" "control_${CONTROL}_256" "${LOG_PATH}" \
            "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
            --task addition --checkpoint "${BACKBONE}" \
            --controller "${SELECTED_CONTROLLER}" \
            --lengths 1 5 10 11 15 20 30 40 --examples 256 \
            --max-batch-size 32 --token-budget 8192 --seed 651001 \
            --device cuda --out-dir "${CONTROL_DIR}" "${CONTROL_ARGS[@]}"
    done

    SPECTRUM_DIR="${OUT_DIR}/selected_spectrum"
    if [[ ! -f "${SPECTRUM_DIR}/spectrum_summary.json" ]]; then
        run_monitored "${JOB}" selected_spectrum "${LOG_PATH}" \
            "${PYTHON_BIN}" -m scripts.analyze_paper_controller_spectrum \
            --artifact "${SELECTED_CONTROLLER}" --label "${JOB}_selected" \
            --out-dir "${SPECTRUM_DIR}"
    fi
    complete_job "${JOB}"
done

ACTIVE_JOB=""; ACTIVE_PHASE=""; ACTIVE_LOG=""
update_manifest complete "" "" "" ""
