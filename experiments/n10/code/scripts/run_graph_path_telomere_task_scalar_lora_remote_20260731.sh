#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL SEED RANK [RANK ...]" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
INITIALIZATION_SEED="$3"
shift 3
RANKS=("$@")

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
PARAMETERIZATION="${PARAMETERIZATION:-scalar_low_rank}"
if [[ "${PARAMETERIZATION}" != "scalar_low_rank" && "${PARAMETERIZATION}" != "diagonal_low_rank" ]]; then
    echo "PARAMETERIZATION must be scalar_low_rank or diagonal_low_rank" >&2
    exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/paperexperiment/graph_path_telomere_task_${PARAMETERIZATION}_j_ceonly_20260731}"
LOG_ROOT=/data/paperexperiment/logs
OUT_DIR="${OUTPUT_ROOT}/${LABEL}"
RUN_LOG="${LOG_ROOT}/task_scalar_lora_${LABEL}.log"
HEARTBEAT_LOG="${LOG_ROOT}/task_scalar_lora_${LABEL}.heartbeat.log"
MANIFEST="${OUT_DIR}/run_manifest.json"
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/paperexperiment/graph_path_telomere_overloop_20260729/phase_grid/D8_L8_seed0/summary.json
REFERENCE_AFFINE=/data/paperexperiment/graph_path_telomere_unit_j_20260731/seed0_curriculum64/unit_j_maps.pt
INITIAL_AFFINE=/data/paperexperiment/graph_path_telomere_unit_j_20260731/seed0_focus_long/unit_j_maps.pt
LR_MULTIPLIER="${LR_MULTIPLIER:-30}"
SCALE_LR_MULTIPLIER="${SCALE_LR_MULTIPLIER:-1}"
DIAGONAL_SCALE_INIT="${DIAGONAL_SCALE_INIT:-diagonal}"

PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
python3 - "${MANIFEST}" "${PHYSICAL_GPU}" "${INITIALIZATION_SEED}" "${LR_MULTIPLIER}" "${SCALE_LR_MULTIPLIER}" "${DIAGONAL_SCALE_INIT}" "${PARAMETERIZATION}" "${RANKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest, gpu, seed, lr, scale_lr, diagonal_init, parameterization, *ranks = sys.argv[1:]
Path(manifest).write_text(json.dumps({
    "status": "launching",
    "parameterization": parameterization,
    "placement": "loop_boundary",
    "identity_scale_init": "trace",
    "learning_rate_multiplier": float(lr),
    "scale_learning_rate_multiplier": float(scale_lr),
    "diagonal_scale_init": diagonal_init,
    "ranks": [int(rank) for rank in ranks],
    "initialization_seeds": [int(seed)],
    "physical_gpu": int(gpu),
}, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"
set +e
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_lora_j \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --reference-affine-artifact "${REFERENCE_AFFINE}" \
    --reference-affine-label task \
    --initialization affine_svd \
    --initial-affine-artifact "${INITIAL_AFFINE}" \
    --initial-affine-label reg \
    --parameterization "${PARAMETERIZATION}" \
    --identity-scale-init trace \
    --diagonal-scale-init "${DIAGONAL_SCALE_INIT}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --placement loop_boundary \
    --ranks "${RANKS[@]}" \
    --initialization-seeds "${INITIALIZATION_SEED}" \
    --learning-rate-multiplier "${LR_MULTIPLIER}" \
    --scale-learning-rate-multiplier "${SCALE_LR_MULTIPLIER}" \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --evaluation-loops 64 \
    --evaluation-seed 212004 \
    --cuda-memory-fraction 0.04 \
    --physical-gpu "${PHYSICAL_GPU}" \
    --prelaunch-used-mib "${PRELAUNCH_USED_MIB}" \
    --prelaunch-free-mib "${PRELAUNCH_FREE_MIB}" \
    --declared-peak-gib 3.0 \
    --reserve-gib 16.0 \
    > "${RUN_LOG}" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -eq 0 ]]; then RUN_STATUS=complete; else RUN_STATUS=failed; fi
python3 - "${MANIFEST}" "${RUN_STATUS}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

path, status, exit_code = sys.argv[1:]
payload = json.loads(Path(path).read_text())
payload.update(status=status, exit_code=int(exit_code))
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
printf '%s label=%s seed=%s gpu=%s status=%s exit=%s\n' \
    "$(date -Is)" "${LABEL}" "${INITIALIZATION_SEED}" "${PHYSICAL_GPU}" \
    "${RUN_STATUS}" "${STATUS}" >> "${HEARTBEAT_LOG}"
exit "${STATUS}"
