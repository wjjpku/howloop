#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL CONTROLLER_DIR" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
CONTROLLER_DIR="$3"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804
BACKBONE="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
SEARCH_ROOT="${SEARCH_ROOT_OVERRIDE:-${RUN_ROOT}/joint_m1to20_j_40k_20260804}"
OUT_DIR="${SEARCH_ROOT}/selection/${LABEL}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/data/paperexperiment/logs/paper_length_telomere_20260731/joint_m1to20_j_40k_20260804}"
LOG_PATH="${LOG_ROOT}/${LABEL}_selection.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
HEARTBEAT_PATH="${OUT_DIR}/heartbeat.json"
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
if [[ ! -f "${BACKBONE}" || ! -d "${CONTROLLER_DIR}" ]]; then
    echo "missing backbone or controller directory" >&2
    exit 3
fi
if [[ -f "${OUT_DIR}/selection.json" ]] && grep -q '"status": "complete"' "${OUT_DIR}/selection.json"; then
    echo "selection already complete: ${OUT_DIR}"
    exit 0
fi

FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < REQUIRED_FREE_MIB )); then
    echo "GPU ${PHYSICAL_GPU} has ${FREE_MIB} MiB free; ${REQUIRED_FREE_MIB} MiB required" >&2
    exit 75
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${LABEL}" "${CONTROLLER_DIR}" "${BACKBONE}" "${FREE_MIB}" <<'PY'
import json
import pathlib
import sys
import time

path, gpu, label, controller_dir, backbone, free = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "status": "launched",
    "created_unix": time.time(),
    "host": "GPU_ARCHIVE_HOST",
    "physical_gpu": int(gpu),
    "label": label,
    "controller_dir": controller_dir,
    "backbone": backbone,
    "validation_lengths": list(range(1, 21)),
    "validation_seed": 1_124_001,
    "examples_per_length": 1_024,
    "id_gate": {"length": 10, "threshold": 0.995},
    "snapshot_stride": 1,
    "selection_objective": "mean supervised-digit exact match over every m=1..20",
    "declared_peak_mib": 2_048,
    "reserve_mib": 16_384,
    "prelaunch_free_mib": int(free),
}, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

COMMAND=(
    "${PYTHON_BIN}" -u scripts/select_addition_lsb_far_j_snapshot.py
    --checkpoint "${BACKBONE}"
    --controller-dir "${CONTROLLER_DIR}"
    --validation-lengths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    --id-gate-length 10
    --id-gate-threshold 0.995
    --batch-size 128
    --batches 8
    --seed 1124001
    --snapshot-stride 1
    --device cuda
    --out-dir "${OUT_DIR}"
)

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

echo "selection complete: ${OUT_DIR}"
