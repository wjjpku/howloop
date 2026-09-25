#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]] || ! [[ "$2" =~ ^[01]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU SHARD_0_OR_1" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
SHARD="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/architecture_round_20260803
SCREEN_ROOT="${RUN_ROOT}/balanced_carry_screen"
ID_SCREEN_ROOT="${RUN_ROOT}/balanced_carry_id1to10_screen"
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
MANIFEST_PATH="${SCREEN_ROOT}/shard${SHARD}_manifest.json"
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

mkdir -p "${SCREEN_ROOT}" "${ID_SCREEN_ROOT}" "${LOG_ROOT}"
FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
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
    "${USED_MIB}" "${FREE_MIB}" "${ACTIVE_PIDS}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "running", "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]), "shard": int(sys.argv[3]),
    "prelaunch_used_mib": int(sys.argv[4]), "prelaunch_free_mib": int(sys.argv[5]),
    "preexisting_pids": sys.argv[6].split(), "declared_peak_mib": 2048,
    "reserve_mib": 16384, "screen_length": 10, "examples": 4096,
    "endpoint_rule": "registered T(n)=n+1", "completed_jobs": [],
}, indent=2, sort_keys=True) + "\n")
PY

update_manifest() {
    local status="$1" job="$2" pid="$3"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${job}" "${pid}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1]); data = json.loads(path.read_text())
data.update({"status": sys.argv[2], "active_job": sys.argv[3] or None,
             "pid": int(sys.argv[4]) if sys.argv[4] else None,
             "updated_unix": time.time()})
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

run_screen() {
    local job="$1" backbone="$2" out_dir="$3" log_path="$4"
    local free_mib child_pid status
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then return 75; fi
    "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${backbone}" --addition-final-carry \
        --lengths 10 --examples 4096 --max-batch-size 64 --token-budget 8192 \
        --seed 501001 --device cuda --out-dir "${out_dir}" > "${log_path}" 2>&1 &
    child_pid=$!; update_manifest running "${job}" "${child_pid}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        sleep 10
    done
    status=0; wait "${child_pid}" || status=$?
    [[ "${status}" -eq 0 ]] || return "${status}"
    complete_job "${job}"
}

run_id_screen() {
    local job="$1" backbone="$2" out_dir="$3" log_path="$4"
    local free_mib child_pid status
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then return 75; fi
    "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${backbone}" --addition-final-carry \
        --lengths {1..10} --examples 1024 --max-batch-size 64 --token-budget 8192 \
        --seed 502001 --device cuda --out-dir "${out_dir}" > "${log_path}" 2>&1 &
    child_pid=$!; update_manifest running "${job}_id1to10" "${child_pid}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            return 76
        fi
        sleep 10
    done
    status=0; wait "${child_pid}" || status=$?
    [[ "${status}" -eq 0 ]] || return "${status}"
    complete_job "${job}_id1to10"
}

for JOB_INDEX in "${!ALL_JOBS[@]}"; do
    if (( JOB_INDEX % 2 != SHARD )); then continue; fi
    JOB="${ALL_JOBS[JOB_INDEX]}"
    BACKBONE="${RUN_ROOT}/backbones/${JOB}/final.pt"
    OUT_DIR="${SCREEN_ROOT}/${JOB}"
    if [[ ! -f "${BACKBONE}" ]]; then
        echo "missing completed backbone ${BACKBONE}" >&2
        exit 4
    fi
    if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
        run_screen "${JOB}" "${BACKBONE}" "${OUT_DIR}" \
            "${LOG_ROOT}/addition_balanced_carry_screen_${JOB}.log"
    else
        complete_job "${JOB}"
    fi
    ID_OUT_DIR="${ID_SCREEN_ROOT}/${JOB}"
    if [[ ! -f "${ID_OUT_DIR}/summary.json" ]]; then
        run_id_screen "${JOB}" "${BACKBONE}" "${ID_OUT_DIR}" \
            "${LOG_ROOT}/addition_balanced_carry_id1to10_${JOB}.log"
    else
        complete_job "${JOB}_id1to10"
    fi
done

# The official 3L8H/T11 10k checkpoints provide the architecture reference.
if [[ "${SHARD}" -eq 0 ]]; then
    for BACKBONE_SEED in 0 1 2; do
        JOB="l3h8t11_seed${BACKBONE_SEED}"
        BACKBONE="/data/wujiaju/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed${BACKBONE_SEED}/checkpoint_010000.pt"
        OUT_DIR="${SCREEN_ROOT}/${JOB}"
        if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
            run_screen "${JOB}" "${BACKBONE}" "${OUT_DIR}" \
                "${LOG_ROOT}/addition_balanced_carry_screen_${JOB}.log"
        else
            complete_job "${JOB}"
        fi
        ID_OUT_DIR="${ID_SCREEN_ROOT}/${JOB}"
        if [[ ! -f "${ID_OUT_DIR}/summary.json" ]]; then
            run_id_screen "${JOB}" "${BACKBONE}" "${ID_OUT_DIR}" \
                "${LOG_ROOT}/addition_balanced_carry_id1to10_${JOB}.log"
        else
            complete_job "${JOB}_id1to10"
        fi
    done
    # Step 40k is the first checkpoint where all three seeds solve the full
    # n=10 Addition answer, so it is the non-undertrained J reference.
    for BACKBONE_SEED in 0 1 2; do
        JOB="l3h8t11_step40k_seed${BACKBONE_SEED}"
        BACKBONE="/data/wujiaju/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed${BACKBONE_SEED}/checkpoint_040000.pt"
        OUT_DIR="${SCREEN_ROOT}/${JOB}"
        if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
            run_screen "${JOB}" "${BACKBONE}" "${OUT_DIR}" \
                "${LOG_ROOT}/addition_balanced_carry_screen_${JOB}.log"
        else
            complete_job "${JOB}"
        fi
        ID_OUT_DIR="${ID_SCREEN_ROOT}/${JOB}"
        if [[ ! -f "${ID_OUT_DIR}/summary.json" ]]; then
            run_id_screen "${JOB}" "${BACKBONE}" "${ID_OUT_DIR}" \
                "${LOG_ROOT}/addition_balanced_carry_id1to10_${JOB}.log"
        else
            complete_job "${JOB}_id1to10"
        fi
    done
fi

update_manifest complete "" ""
