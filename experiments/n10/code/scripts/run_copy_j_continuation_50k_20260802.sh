#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
BACKBONE="${RUN_ROOT}/backbones/copy_adaptive_step_official_seed0/final.pt"
SOURCE_LABEL=copy_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_postfinal_wsd_lr3em4_warmup2048_stable2816_5k_seed211001
SOURCE_CONTROLLER="${RUN_ROOT}/controllers/${SOURCE_LABEL}/controller.pt"
CONTINUATION_LABEL=copy_adaptive_step_official_seed0_rank48_anchor1_postfinal_continue5376to50000_mix20_30_50_wsd5k_35k_4624_lr3em4_seed311001
OUT_DIR="${RUN_ROOT}/controllers/${CONTINUATION_LABEL}"
LOG_PATH="${LOG_ROOT}/${CONTINUATION_LABEL}.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
HEARTBEAT_PATH="${OUT_DIR}/heartbeat.json"
DECLARED_PEAK_MIB="${DECLARED_PEAK_MIB:-3200}"
RESERVE_MIB="${RESERVE_MIB:-16384}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" || ! -f "${SOURCE_CONTROLLER}" ]]; then
    echo "missing backbone or source controller" >&2
    exit 2
fi

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
if [[ -n "${ACTIVE_PIDS}" || "${PRELAUNCH_USED_MIB}" -gt 128 ]]; then
    if [[ "${ALLOW_SHARED_GPU}" != "1" ]]; then
        echo "refusing non-empty GPU ${PHYSICAL_GPU}" >&2
        exit 75
    fi
    if [[ "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "shared GPU ${PHYSICAL_GPU} has ${PRELAUNCH_FREE_MIB}MiB free; ${REQUIRED_FREE_MIB}MiB required" >&2
        exit 75
    fi
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
    "${PYTHON_BIN}" -m reasoning_loop.continue_paper_length_controller
    --checkpoint "${BACKBONE}"
    --source-controller "${SOURCE_CONTROLLER}"
    --start-total-update 5376
    --target-total-update 50000
    --batch-size 32
    --retention-probability 0.20
    --transition-probability 0.30
    --boundary-probability 0.50
    --peak-learning-rate 3e-4
    --diagonal-lr-multiplier 0.1
    --warmup-updates 5000
    --stable-updates 35000
    --final-learning-rate-ratio 0.1
    --grad-clip 1.0
    --seed 311001
    --eval-seed 361001
    --eval-lengths 1 19 25 32 35 38 40 50
    --eval-batch-size 128
    --eval-batches 2
    --eval-every 1000
    --log-every 100
    --checkpoint-every 5000
    --device cuda
    --out-dir "${OUT_DIR}"
    "${RESUME_ARGS[@]}"
)

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<PY
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "created_unix": time.time(),
    "physical_gpu": ${PHYSICAL_GPU},
    "prelaunch_used_mib": ${PRELAUNCH_USED_MIB},
    "prelaunch_free_mib": ${PRELAUNCH_FREE_MIB},
    "prelaunch_utilization_percent": ${PRELAUNCH_UTILIZATION},
    "active_pids": "${ACTIVE_PIDS//$'\n'/ }".split(),
    "allow_shared_gpu": bool(${ALLOW_SHARED_GPU}),
    "declared_peak_mib": ${DECLARED_PEAK_MIB},
    "reserve_mib": ${RESERVE_MIB},
    "backbone": "${BACKBONE}",
    "source_controller": "${SOURCE_CONTROLLER}",
    "optimizer_resume_mode": "fresh_adamw_from_source_weights_then_exact_resume",
    "start_total_update": 5376,
    "target_total_update": 50000,
    "additional_wsd": {"warmup": 5000, "stable": 35000, "decay": 4624},
    "length_mix": {"1-19": 0.20, "20-32": 0.30, "33-40": 0.50},
    "peak_learning_rate": 3e-4,
    "final_learning_rate": 3e-5,
    "batch_size": 32,
    "loss": "final registered T(n) answer-region CE only",
    "resume": "${RESUME_ARGS[*]:-none}",
}, indent=2, sort_keys=True) + "\n")
PY

printf 'launch_command=' >> "${LOG_PATH}"
printf '%q ' "${COMMAND[@]}" >> "${LOG_PATH}"
printf '\n' >> "${LOG_PATH}"
"${COMMAND[@]}" >> "${LOG_PATH}" 2>&1 &
CHILD_PID=$!

while kill -0 "${CHILD_PID}" 2>/dev/null; do
    USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" <<PY
import json, pathlib, time, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(), "child_pid": ${CHILD_PID},
    "physical_gpu": ${PHYSICAL_GPU}, "used_mib": ${USED_MIB},
    "free_mib": ${FREE_MIB}, "utilization_percent": ${UTILIZATION}
}, indent=2, sort_keys=True) + "\n")
PY
    if [[ "${FREE_MIB}" -lt "${RESERVE_MIB}" ]]; then
        echo "reserve guard triggered: free=${FREE_MIB}MiB reserve=${RESERVE_MIB}MiB; stopping child ${CHILD_PID}" >> "${LOG_PATH}"
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" || true
        exit 76
    fi
    sleep 30
done

wait "${CHILD_PID}"
