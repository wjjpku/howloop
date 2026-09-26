#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]] || [[ "$#" -gt 3 ]] || \
   { [[ "$1" != "armA" ]] && [[ "$1" != "armB" ]]; } || \
   ! [[ "$2" =~ ^[0-7]$ ]] || \
   { [[ "$#" -eq 3 ]] && ! [[ "$3" =~ ^[0-9]+$ ]]; }; then
    echo "usage: $0 {armA|armB} PHYSICAL_GPU [BACKBONE_SEED]" >&2
    exit 2
fi

ARM="$1"
PHYSICAL_GPU="$2"
BACKBONE_SEED="${3:-0}"
if [[ "${ARM}" == "armA" ]] && [[ "${BACKBONE_SEED}" -ne 0 ]]; then
    echo "armA is the historical fixed-n10 seed0 checkpoint; use armB for multiseed runs" >&2
    exit 2
fi
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
SOURCE_ROOT=/data/paperexperiment/paper_length_telomere_20260731
RUN_ROOT="${SOURCE_ROOT}/full_answer_redesign_20260803"
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731/full_answer_redesign_20260803
RESERVE_MIB=16384
DECLARED_PEAK_MIB=6144
REQUIRED_FREE_MIB=$((RESERVE_MIB + DECLARED_PEAK_MIB))

if [[ "${ARM}" == "armA" ]]; then
    LABEL=armA_fixed_n10_40k_seed0
    BACKBONE="${SOURCE_ROOT}/backbones/addition_fixed_n10_t11_official_seed0/checkpoint_040000.pt"
else
    LABEL="armB_adaptive_n1to10_40k_seed${BACKBONE_SEED}"
    BACKBONE_DIR="${RUN_ROOT}/${LABEL}/backbone"
    BACKBONE="${BACKBONE_DIR}/final.pt"
fi
ARM_ROOT="${RUN_ROOT}/${LABEL}"
CONTROLLER_DIR="${ARM_ROOT}/controller_rank48_identity_fullanswer_l1to10_anchor1_wsd5376"
CONTROLLER="${CONTROLLER_DIR}/controller.pt"
MANIFEST_PATH="${ARM_ROOT}/pipeline_manifest.json"
HEARTBEAT_PATH="${ARM_ROOT}/heartbeat.json"
mkdir -p "${ARM_ROOT}" "${LOG_ROOT}"

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

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${ARM}" "${LABEL}" \
    "${PHYSICAL_GPU}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" "${BACKBONE}" \
    "${BACKBONE_SEED}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "arm": sys.argv[2],
    "label": sys.argv[3],
    "physical_gpu": int(sys.argv[4]),
    "prelaunch_used_mib": int(sys.argv[5]),
    "prelaunch_free_mib": int(sys.argv[6]),
    "prelaunch_utilization_percent": int(sys.argv[7]),
    "preexisting_pids": sys.argv[8].split(),
    "backbone": sys.argv[9],
    "backbone_seed": int(sys.argv[10]),
    "backbone_training": (
        "existing official fixed-n10 checkpoint at update 40000"
        if sys.argv[2] == "armA"
        else "official Addition model, adaptive full-answer CE on logical lengths 1..10, T(k)=k+1, update 40000"
    ),
    "controller": "J(h)=hD+(hA)B+b, rank 48, shared before loops 2 onward",
    "controller_training_logical_range": [1, 10],
    "controller_loss": "full answer-region CE only at registered T(k)=k+1",
    "controller_initialization": "exact identity",
    "controller_optimizer_updates": 5376,
    "controller_snapshots": "update 0, every 256 updates, and final update",
    "controller_wsd": {"warmup": 2048, "stable": 2816, "decay": 512},
    "controller_peak_learning_rate": 1e-4,
    "selection": "ID lengths 1..10 only; require per-length EM>=0.99, then minimize mean answer CE",
    "strict_unseen_logical_range": [11, 40],
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
pathlib.Path(sys.argv[1]).write_text(json.dumps({"unix": time.time(), "phase": sys.argv[2], "child_pid": int(sys.argv[3]), "physical_gpu": int(sys.argv[4]), "used_mib": int(sys.argv[5]), "free_mib": int(sys.argv[6]), "utilization_percent": int(sys.argv[7])}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            echo "reserve guard triggered in ${phase}: free=${free_mib}MiB" >> "${log_path}"
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        # Long phases still emit a heartbeat every ~30 seconds, but short
        # snapshot evaluations should hand control back within two seconds.
        for _ in {1..15}; do
            sleep 2
            if ! kill -0 "${child_pid}" 2>/dev/null; then break; fi
        done
    done
    status=0
    wait "${child_pid}" || status=$?
    if [[ "${status}" -eq 0 ]]; then complete_phase "${phase}"; fi
    return "${status}"
}

if [[ "${ARM}" == "armB" ]] && [[ ! -f "${BACKBONE}" ]]; then
    mkdir -p "${BACKBONE_DIR}"
    run_monitored train_matched_backbone_40k "${LOG_ROOT}/${LABEL}_backbone.log" \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone \
        --task addition --supervision adaptive_step --steps 40001 \
        --schedule-total-steps 100001 --batch-size 64 --learning-rate 1e-4 \
        --weight-decay 0.01 --grad-clip 1.0 --official-model-config \
        --train-max-length 10 --curriculum-interval 1000 \
        --seed "${BACKBONE_SEED}" \
        --no-amp --device cuda --log-every 100 --eval-every 1000 \
        --eval-batch-size 256 --eval-batches 4 --checkpoint-every 5000 \
        --out-dir "${BACKBONE_DIR}"
else
    complete_phase matched_backbone_available
fi
if [[ ! -f "${BACKBONE}" ]]; then
    echo "missing backbone ${BACKBONE}" >&2
    exit 3
fi

RAW_DIR="${ARM_ROOT}/raw_fullanswer_l1to40"
if [[ ! -f "${RAW_DIR}/summary.json" ]]; then
    RAW_LENGTHS=()
    for ((length=1; length<=40; length+=1)); do RAW_LENGTHS+=("${length}"); done
    run_monitored raw_eval_l1to40 "${LOG_ROOT}/${LABEL}_raw_l1to40.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${BACKBONE}" --lengths "${RAW_LENGTHS[@]}" \
        --examples 256 --max-batch-size 32 --token-budget 8192 \
        --seed 511001 --device cuda --out-dir "${RAW_DIR}"
else
    complete_phase raw_eval_l1to40
fi

if [[ ! -f "${CONTROLLER}" ]]; then
    run_monitored train_fullanswer_j_5376 "${LOG_ROOT}/${LABEL}_controller.log" \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
        --checkpoint "${BACKBONE}" --rank 48 \
        --seed "$((521001 + BACKBONE_SEED))" \
        --controller-curriculum logical_range --controller-logical-min-length 1 \
        --controller-logical-max-length 10 \
        --controller-training-profile identity_long_warmup \
        --controller-initialization identity --controller-anchor-step 1 \
        --controller-warmup-updates 2048 --controller-final-lr-ratio 0.1 \
        --controller-lr-schedule wsd --controller-stable-updates 2816 \
        --controller-supervision full_answer --controller-checkpoint-every 256 \
        --dense-stage-count 0 --grad-clip 1.0 \
        --learning-rate-multiplier 5.0 --diagonal-lr-multiplier 0.1 \
        --stage-round-multiplier 3 --force --device cuda \
        --out-dir "${CONTROLLER_DIR}"
else
    complete_phase train_fullanswer_j_5376
fi

ID_ROOT="${ARM_ROOT}/snapshot_id_eval_l1to10"
mkdir -p "${ID_ROOT}"
for SNAPSHOT in "${CONTROLLER_DIR}"/checkpoints/controller_*.pt; do
    SNAPSHOT_LABEL="$(basename "${SNAPSHOT}" .pt)"
    SNAPSHOT_DIR="${ID_ROOT}/${SNAPSHOT_LABEL}"
    if [[ -f "${SNAPSHOT_DIR}/summary.json" ]]; then continue; fi
    run_monitored "id_eval_${SNAPSHOT_LABEL}" \
        "${LOG_ROOT}/${LABEL}_${SNAPSHOT_LABEL}_id.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${BACKBONE}" --controller "${SNAPSHOT}" \
        --lengths 1 2 3 4 5 6 7 8 9 10 --examples 128 \
        --max-batch-size 32 --token-budget 8192 --seed 531001 --device cuda \
        --out-dir "${SNAPSHOT_DIR}"
done
complete_phase snapshot_id_eval_l1to10

SELECTION="${ARM_ROOT}/snapshot_selection.json"
"${PYTHON_BIN}" scripts/select_controller_snapshot.py \
    --evaluation-root "${ID_ROOT}" --lengths 1 2 3 4 5 6 7 8 9 10 \
    --minimum-exact-match 0.99 --out "${SELECTION}" \
    >> "${LOG_ROOT}/${LABEL}_selection.log" 2>&1
complete_phase select_snapshot_from_id_only
SELECTED_CONTROLLER="$("${PYTHON_BIN}" - "${SELECTION}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected"]["controller"])
PY
)"

SELECTED_DIR="${ARM_ROOT}/selected_fullanswer_l1to40_512"
if [[ ! -f "${SELECTED_DIR}/summary.json" ]]; then
    SELECTED_LENGTHS=()
    for ((length=1; length<=40; length+=1)); do SELECTED_LENGTHS+=("${length}"); done
    run_monitored selected_eval_l1to40_512 "${LOG_ROOT}/${LABEL}_selected_l1to40.log" \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${BACKBONE}" \
        --controller "${SELECTED_CONTROLLER}" --lengths "${SELECTED_LENGTHS[@]}" \
        --examples 512 --max-batch-size 32 --token-budget 8192 \
        --seed 541001 --device cuda --out-dir "${SELECTED_DIR}"
else
    complete_phase selected_eval_l1to40_512
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${SELECTED_CONTROLLER}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "active_phase": None,
    "pid": None,
    "selected_controller": sys.argv[2],
    "finished_unix": time.time(),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
