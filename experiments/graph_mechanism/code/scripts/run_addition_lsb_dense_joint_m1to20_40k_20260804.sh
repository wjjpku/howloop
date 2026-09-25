#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804
BACKBONE="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
SEARCH_ROOT="${RUN_ROOT}/dense_joint_m1to20_40k_20260804"
OUT_DIR="${SEARCH_ROOT}/training/${LABEL}"
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731/dense_joint_m1to20_40k_20260804
LOG_PATH="${LOG_ROOT}/${LABEL}.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
HEARTBEAT_PATH="${OUT_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" ]]; then
    echo "missing backbone: ${BACKBONE}" >&2
    exit 3
fi
if [[ -f "${OUT_DIR}/summary.json" ]] && grep -q '"status": "complete"' "${OUT_DIR}/summary.json"; then
    echo "dense training already complete: ${OUT_DIR}"
    exit 0
fi

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if (( PRELAUNCH_FREE_MIB < REQUIRED_FREE_MIB )); then
    echo "GPU ${PHYSICAL_GPU} has ${PRELAUNCH_FREE_MIB} MiB free; ${REQUIRED_FREE_MIB} MiB required" >&2
    exit 75
fi

RESUME_ARGS=()
if [[ -f "${OUT_DIR}/latest.pt" ]]; then
    RESUME_ARGS=(--resume "${OUT_DIR}/latest.pt")
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

COMMAND=(
    "${PYTHON_BIN}" -u -m reasoning_loop.continue_paper_length_controller
    --task addition
    --checkpoint "${BACKBONE}"
    --initialize-dense-identity
    --anchor-step 1
    --start-total-update 0
    --target-total-update 40000
    --batch-size 32
    --minimum-length 1
    --retention-maximum 10
    --transition-maximum 15
    --boundary-maximum 20
    --retention-probability 0.5
    --transition-probability 0.25
    --boundary-probability 0.25
    --peak-learning-rate 6e-5
    --diagonal-lr-multiplier 0.1
    --warmup-updates 4096
    --stable-updates 31232
    --final-learning-rate-ratio 0.1
    --grad-clip 1.0
    --seed 921101
    --eval-seed 1024001
    --eval-lengths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    --selection-lengths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    --selection-mode mean_ce
    --eval-batch-size 128
    --eval-batches 2
    --eval-every 1024
    --log-every 128
    --checkpoint-every 2048
    --device cuda
    --out-dir "${OUT_DIR}"
    "${RESUME_ARGS[@]}"
)

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${LABEL}" \
    "${BACKBONE}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json
import pathlib
import sys
import time

path, gpu, label, backbone, used, free, utilization, active_pids = sys.argv[1:]
payload = {
    "status": "launched",
    "created_unix": time.time(),
    "host": "A100-80G-34200",
    "physical_gpu": int(gpu),
    "label": label,
    "backbone": backbone,
    "controller": "J(h)=hW+b; all 65,792 W/b parameters trainable",
    "controller_parameterization": "dense_affine",
    "controller_initialization": "exact_identity",
    "anchor_step": 1,
    "target_total_updates": 40_000,
    "wsd": {"warmup": 4_096, "stable": 31_232, "decay": 4_672},
    "logical_training_lengths": [1, 20],
    "length_sampling": "uniform over every integer m=1..20",
    "structural_noop_lengths": [1],
    "structural_noop_reason": "T(1)=1 has no inter-loop J application; evaluated and counted but cannot update J",
    "expected_gradient_updates": 38_000,
    "loss": "final-only supervised-digit CE at T(m)=m",
    "controller_seed": 921_101,
    "evaluation_seed": 1_024_001,
    "declared_peak_mib": 2_048,
    "matching_dense_peak_reserved_mib": 1_318,
    "reserve_mib": 16_384,
    "prelaunch_used_mib": int(used),
    "prelaunch_free_mib": int(free),
    "prelaunch_utilization_percent": int(utilization),
    "preexisting_pids": active_pids.split(),
    "expected_storage_mib": 30,
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf 'launch_command=' > "${LOG_PATH}"
printf '%q ' "${COMMAND[@]}" >> "${LOG_PATH}"
printf '\n' >> "${LOG_PATH}"
"${COMMAND[@]}" >> "${LOG_PATH}" 2>&1 &
CHILD_PID=$!

while kill -0 "${CHILD_PID}" 2>/dev/null; do
    USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${CHILD_PID}" "${PHYSICAL_GPU}" "${USED_MIB}" "${FREE_MIB}" <<'PY'
import json
import pathlib
import sys
import time

path, child, gpu, used, free = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "unix": time.time(),
    "child_pid": int(child),
    "physical_gpu": int(gpu),
    "used_mib": int(used),
    "free_mib": int(free),
}, indent=2, sort_keys=True) + "\n")
PY
    if (( FREE_MIB < RESERVE_MIB )); then
        echo "reserve guard triggered: free=${FREE_MIB} MiB reserve=${RESERVE_MIB} MiB; stopping child ${CHILD_PID}" >> "${LOG_PATH}"
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" || true
        exit 76
    fi
    sleep 30
done

wait "${CHILD_PID}"
"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["status"] = "complete"
payload["completed_unix"] = time.time()
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "dense training complete: ${OUT_DIR}"
