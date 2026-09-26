#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 8 ]]; then
    echo "usage: $0 GPU RUN_LABEL CHECKPOINT PHASE_JSON BACKBONE_LOSS STATE_WEIGHT MAX_HORIZON SEED [SEED ...]" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
RUN_LABEL="$2"
CHECKPOINT="$3"
PHASE_SUMMARY="$4"
BACKBONE_LOSS="$5"
STATE_LOSS_WEIGHT="$6"
MAX_TRAINING_HORIZON="$7"
shift 7
INITIALIZATION_SEEDS=("$@")
RANK="${RANK:-48}"
INITIALIZATION_MODE="${INITIALIZATION_MODE:-affine_svd}"
SHARED_GPU="${SHARED_GPU:-0}"
DECLARED_PEAK_GIB="${DECLARED_PEAK_GIB:-4.0}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.05}"
RESERVE_GIB="${RESERVE_GIB:-16.0}"
PLACEMENT="${PLACEMENT:-loop_boundary}"
if [[ "${PLACEMENT}" != "loop_boundary" && "${PLACEMENT}" != "pre_block2" ]]; then
    echo "PLACEMENT must be loop_boundary or pre_block2" >&2
    exit 2
fi
if [[ "${INITIALIZATION_MODE}" != "affine_svd" && "${INITIALIZATION_MODE}" != "identity" ]]; then
    echo "INITIALIZATION_MODE must be affine_svd or identity" >&2
    exit 2
fi

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
OUTPUT_ROOT=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731
LOG_ROOT=/data/paperexperiment/logs/graph_path_telomere_canonical_diag_lora_20260731
INITIALIZER_DIR="${OUTPUT_ROOT}/initializers/${RUN_LABEL}"
OUT_DIR="${OUTPUT_ROOT}/controllers/${RUN_LABEL}"
RUN_LOG="${LOG_ROOT}/${RUN_LABEL}.log"
HEARTBEAT_LOG="${LOG_ROOT}/${RUN_LABEL}.heartbeat.log"
MANIFEST="${OUT_DIR}/run_manifest.json"

mkdir -p "${INITIALIZER_DIR}" "${OUT_DIR}" "${LOG_ROOT}"
PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
REQUIRED_FREE_MIB="$(awk -v peak="${DECLARED_PEAK_GIB}" -v reserve="${RESERVE_GIB}" 'BEGIN { printf "%d", (peak + reserve) * 1024 }')"
if [[ "${SHARED_GPU}" == "1" && "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "shared-GPU launch rejected: free=${PRELAUNCH_FREE_MIB}MiB required=${REQUIRED_FREE_MIB}MiB" >&2
    exit 75
fi

"${PYTHON_BIN}" - "${MANIFEST}" "${RUN_LABEL}" "${PHYSICAL_GPU}" \
    "${CHECKPOINT}" "${PHASE_SUMMARY}" "${BACKBONE_LOSS}" \
    "${STATE_LOSS_WEIGHT}" "${MAX_TRAINING_HORIZON}" "${RANK}" "${INITIALIZATION_MODE}" \
    "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" "${SHARED_GPU}" "${DECLARED_PEAK_GIB}" \
    "${PLACEMENT}" \
    "${INITIALIZATION_SEEDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest,
    label,
    gpu,
    checkpoint,
    phase,
    backbone_loss,
    state_weight,
    max_horizon,
    rank,
    initialization_mode,
    used,
    free,
    shared,
    declared_peak,
    placement,
    *seeds,
) = sys.argv[1:]
Path(manifest).write_text(json.dumps({
    "status": "launching",
    "run_label": label,
    "checkpoint": checkpoint,
    "phase_summary": phase,
    "backbone_loss_description": backbone_loss,
    "controller": "J(h)=hD+(hA)B+b",
    "parameterization": "diagonal_low_rank",
    "rank": int(rank),
    "placement": placement,
    "controller_loss": "phase-matched task CE at every controlled step" + (
        "; no hidden-state loss"
        if float(state_weight) == 0.0
        else f"; normalized hidden-state MSE weight {float(state_weight):g}"
    ),
    "state_loss_weight": float(state_weight),
    "max_training_horizon": int(max_horizon),
    "initialization": (
        f"{placement} explicit diagonal reduced-rank ridge r{rank}"
        if initialization_mode == "affine_svd"
        else "exact identity D=I, AB=0, b=0"
    ),
    "initialization_mode": initialization_mode,
    "initialization_seeds": [int(seed) for seed in seeds],
    "physical_gpu": int(gpu),
    "prelaunch_used_mib": int(used),
    "prelaunch_free_mib": int(free),
    "shared_gpu": bool(int(shared)),
    "declared_peak_gib": float(declared_peak),
}, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

set +e
{
    if [[ ! -f "${INITIALIZER_DIR}/unit_j_maps.pt" ]]; then
        "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_explicit_diagonal_rrr \
            --checkpoint "${CHECKPOINT}" \
            --phase-summary "${PHASE_SUMMARY}" \
            --out-dir "${INITIALIZER_DIR}" \
            --device cuda \
            --placement "${PLACEMENT}" \
            --backbone-loss-description "${BACKBONE_LOSS}" \
            --graphs 1024 \
            --batch-size 32 \
            --data-seed 20260731 \
            --answer-weight 28 \
            --identity-weight 1 \
            --ridge 0.01 \
            --iterations 12 \
            --diagonal-init dense_diagonal \
            --ranks "${RANK}" \
            --evaluation-batch-size 128 \
            --evaluation-batches 4 \
            --evaluation-loops 64 \
            --evaluation-seed 212004
    fi

    INITIALIZATION_ARGS=(--initialization "${INITIALIZATION_MODE}")
    if [[ "${INITIALIZATION_MODE}" == "affine_svd" ]]; then
        INITIALIZATION_ARGS+=(
            --initial-affine-artifact "${INITIALIZER_DIR}/unit_j_maps.pt"
            --initial-affine-label "explicit_diag_rrr_r${RANK}"
        )
    fi
    SHARED_ARGS=()
    if [[ "${SHARED_GPU}" == "1" ]]; then
        SHARED_ARGS+=(--shared-gpu)
    fi

    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_lora_j \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --reference-affine-artifact "${INITIALIZER_DIR}/unit_j_maps.pt" \
        --reference-affine-label "explicit_diag_rrr_r${RANK}" \
        "${INITIALIZATION_ARGS[@]}" \
        --require-affine-placement \
        --parameterization diagonal_low_rank \
        --diagonal-scale-init diagonal \
        --out-dir "${OUT_DIR}" \
        --device cuda \
        --placement "${PLACEMENT}" \
        --ranks "${RANK}" \
        --initialization-seeds "${INITIALIZATION_SEEDS[@]}" \
        --learning-rate-multiplier 30 \
        --scale-learning-rate-multiplier 0.1 \
        --state-loss-weight "${STATE_LOSS_WEIGHT}" \
        --max-training-horizon "${MAX_TRAINING_HORIZON}" \
        --backbone-loss-description "${BACKBONE_LOSS}" \
        --evaluation-batch-size 128 \
        --evaluation-batches 8 \
        --evaluation-loops 128 \
        --evaluation-seed 212004 \
        --cuda-memory-fraction "${CUDA_MEMORY_FRACTION}" \
        --physical-gpu "${PHYSICAL_GPU}" \
        --prelaunch-used-mib "${PRELAUNCH_USED_MIB}" \
        --prelaunch-free-mib "${PRELAUNCH_FREE_MIB}" \
        --declared-peak-gib "${DECLARED_PEAK_GIB}" \
        --reserve-gib "${RESERVE_GIB}" \
        "${SHARED_ARGS[@]}"
} > "${RUN_LOG}" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
"${PYTHON_BIN}" - "${MANIFEST}" "${RUN_STATUS}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

path, status, exit_code = sys.argv[1:]
payload = json.loads(Path(path).read_text())
payload.update(status=status, exit_code=int(exit_code))
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
printf '%s label=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${RUN_LABEL}" "${PHYSICAL_GPU}" "${RUN_STATUS}" "${STATUS}" \
    >> "${HEARTBEAT_LOG}"
exit "${STATUS}"
