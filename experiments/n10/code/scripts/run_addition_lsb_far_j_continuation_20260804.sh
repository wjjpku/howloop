#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL SOURCE_CONTROLLER" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
SOURCE_CONTROLLER="$3"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804
BACKBONE="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
SEARCH_ROOT="${RUN_ROOT}/far_j_longtrain_20260804"
OUT_DIR="${SEARCH_ROOT}/continuations/${LABEL}"
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731/far_j_longtrain_20260804
LOG_PATH="${LOG_ROOT}/${LABEL}.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
HEARTBEAT_PATH="${OUT_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" || ! -f "${SOURCE_CONTROLLER}" ]]; then
    echo "missing backbone or source controller" >&2
    exit 3
fi
if [[ -f "${OUT_DIR}/summary.json" ]] && grep -q '"status": "complete"' "${OUT_DIR}/summary.json"; then
    echo "continuation already complete: ${OUT_DIR}"
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

SOURCE_METADATA="$("${PYTHON_BIN}" - "${SOURCE_CONTROLLER}" "${BACKBONE}" <<'PY'
import json
import pathlib
import sys
import torch

source = pathlib.Path(sys.argv[1])
payload = torch.load(source, map_location="cpu", weights_only=False)
if payload.get("kind") != "paper_length_telomere_controller":
    raise SystemExit("source is not a controller artifact")
if payload.get("controller_parameterization") != "diagonal_low_rank":
    raise SystemExit("source must be diagonal_low_rank")
if payload.get("checkpoint") != sys.argv[2]:
    raise SystemExit("source backbone mismatch")
if payload.get("task", {}).get("name") != "addition":
    raise SystemExit("source task is not Addition")
if payload.get("controller_post_final_j") is not False:
    raise SystemExit("source unexpectedly uses post-final J")
if int(payload["training_budget"]["total_optimizer_updates"]) != 5376:
    raise SystemExit("source does not start at 5376 updates")
print(json.dumps({
    "rank": int(payload["rank"]),
    "source_seed": int(payload["seed"]),
    "source_updates": int(payload["training_budget"]["total_optimizer_updates"]),
}))
PY
)"

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
    --source-controller "${SOURCE_CONTROLLER}"
    --start-total-update 5376
    --target-total-update 20000
    --batch-size 32
    --minimum-length 10
    --retention-maximum 13
    --transition-maximum 17
    --boundary-maximum 20
    --retention-probability 0.36363636363636365
    --transition-probability 0.36363636363636365
    --boundary-probability 0.2727272727272727
    --peak-learning-rate 6e-5
    --diagonal-lr-multiplier 0.1
    --warmup-updates 2048
    --stable-updates 10240
    --final-learning-rate-ratio 0.1
    --grad-clip 1.0
    --seed 721101
    --eval-seed 824001
    --eval-lengths 10 11 12 13 14 15 16 17 18 19 20
    --selection-lengths 10 11 12 13 14 15 16 17 18 19 20
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
    "${SOURCE_CONTROLLER}" "${BACKBONE}" "${SOURCE_METADATA}" \
    "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" \
    "${PRELAUNCH_UTILIZATION}" "${ACTIVE_PIDS}" <<'PY'
import json
import pathlib
import sys
import time

(
    path, gpu, label, source, backbone, source_metadata,
    used, free, utilization, active_pids,
) = sys.argv[1:]
payload = {
    "status": "launched",
    "created_unix": time.time(),
    "host": "GPU_ARCHIVE_HOST",
    "physical_gpu": int(gpu),
    "label": label,
    "backbone": backbone,
    "source_controller": source,
    "source": json.loads(source_metadata),
    "optimizer_resume_mode": "fresh AdamW from selected 5376-step weights",
    "target_total_updates": 20000,
    "additional_updates": 14624,
    "additional_wsd": {"warmup": 2048, "stable": 10240, "decay": 2336},
    "logical_training_lengths": [10, 20],
    "length_sampling": "uniform over integers 10..20 via size-weighted bands",
    "loss": "final-only supervised-digit CE at T(m)=m",
    "controller_seed": 721101,
    "evaluation_seed": 824001,
    "declared_peak_mib": 2048,
    "reserve_mib": 16384,
    "prelaunch_used_mib": int(used),
    "prelaunch_free_mib": int(free),
    "prelaunch_utilization_percent": int(utilization),
    "preexisting_pids": active_pids.split(),
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
    UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${CHILD_PID}" "${PHYSICAL_GPU}" \
        "${USED_MIB}" "${FREE_MIB}" "${UTILIZATION}" <<'PY'
import json
import pathlib
import sys
import time

path, child, gpu, used, free, utilization = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "unix": time.time(),
    "child_pid": int(child),
    "physical_gpu": int(gpu),
    "used_mib": int(used),
    "free_mib": int(free),
    "utilization_percent": int(utilization),
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

echo "continuation complete: ${OUT_DIR}"
