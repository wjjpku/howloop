#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL PARAMETERIZATION RANK LR_MULTIPLIER MIN_LENGTH MAX_LENGTH CONTROLLER_SEED" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
PARAMETERIZATION="$3"
RANK="$4"
LR_MULTIPLIER="$5"
MIN_LENGTH="$6"
MAX_LENGTH="$7"
CONTROLLER_SEED="$8"

if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
    echo "invalid physical GPU: ${PHYSICAL_GPU}" >&2
    exit 2
fi
if [[ "${PARAMETERIZATION}" != "diagonal_low_rank" && "${PARAMETERIZATION}" != "dense_affine" ]]; then
    echo "invalid parameterization: ${PARAMETERIZATION}" >&2
    exit 2
fi

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804
CHECKPOINT="${RUN_ROOT}/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
SEARCH_ROOT="${RUN_ROOT}/far_j_search_20260804"
OUT_DIR="${SEARCH_ROOT}/controllers/${LABEL}"
LOG_DIR=/data/paperexperiment/logs/paper_length_telomere_20260731/far_j_search_20260804
LOG_PATH="${LOG_DIR}/${LABEL}.log"
MANIFEST_PATH="${OUT_DIR}/launch_manifest.json"
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ -f "${OUT_DIR}/summary.json" ]] && grep -q '"status": "complete"' "${OUT_DIR}/summary.json"; then
    echo "candidate already complete: ${OUT_DIR}"
    exit 0
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing backbone checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi

FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < REQUIRED_FREE_MIB )); then
    echo "GPU ${PHYSICAL_GPU} has ${FREE_MIB} MiB free; ${REQUIRED_FREE_MIB} MiB required" >&2
    exit 75
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${LABEL}" \
    "${PARAMETERIZATION}" "${RANK}" "${LR_MULTIPLIER}" "${MIN_LENGTH}" \
    "${MAX_LENGTH}" "${CONTROLLER_SEED}" "${CHECKPOINT}" <<'PY'
import json
import pathlib
import sys
import time

(
    manifest,
    gpu,
    label,
    parameterization,
    rank,
    lr_multiplier,
    minimum,
    maximum,
    seed,
    checkpoint,
) = sys.argv[1:]
payload = {
    "status": "launched",
    "created_unix": time.time(),
    "host": "GPU_ARCHIVE_HOST",
    "physical_gpu": int(gpu),
    "label": label,
    "checkpoint": checkpoint,
    "checkpoint_step": 80000,
    "behavior": "Addition LSB-to-MSB, causal attention, no position embedding",
    "backbone_training_lengths": [1, 10],
    "backbone_loss_placement": "final-only supervised-digit CE at T(m)=m",
    "shared_physical_block_layers": 3,
    "controller": {
        "parameterization": parameterization,
        "rank": int(rank) if parameterization == "diagonal_low_rank" else None,
        "initialization": "identity",
        "logical_training_lengths": [int(minimum), int(maximum)],
        "anchor_step": 1,
        "post_final_j": False,
        "loss": "final-only supervised-digit CE at T(m)=m",
        "learning_rate_multiplier": float(lr_multiplier),
        "warmup_updates": 2048,
        "stable_updates": 2816,
        "schedule": "wsd",
        "stage_round_multiplier": 3,
        "checkpoint_every": 256,
        "seed": int(seed),
    },
    "declared_peak_mib": 2048,
    "reserve_mib": 16384,
}
path = pathlib.Path(manifest)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

COMMAND=(
    "${PYTHON_BIN}" -u -m reasoning_loop.paper_length_telomere controller
    --checkpoint "${CHECKPOINT}"
    --controller-parameterization "${PARAMETERIZATION}"
    --seed "${CONTROLLER_SEED}"
    --device cuda
    --grad-clip 1.0
    --learning-rate-multiplier "${LR_MULTIPLIER}"
    --diagonal-lr-multiplier 0.1
    --dense-stage-count 0
    --controller-training-profile identity_long_warmup
    --controller-initialization identity
    --controller-curriculum logical_range
    --controller-logical-min-length "${MIN_LENGTH}"
    --controller-logical-max-length "${MAX_LENGTH}"
    --controller-anchor-step 1
    --controller-warmup-updates 2048
    --controller-stable-updates 2816
    --controller-final-lr-ratio 0.1
    --controller-lr-schedule wsd
    --controller-ce-temperature 1.0
    --controller-supervision full_answer
    --stage-round-multiplier 3
    --controller-checkpoint-every 256
    --no-controller-post-final-j
    --force
    --out-dir "${OUT_DIR}"
)
if [[ "${PARAMETERIZATION}" == "diagonal_low_rank" ]]; then
    COMMAND+=(--rank "${RANK}")
fi

printf 'launch_command=' > "${LOG_PATH}"
printf '%q ' "${COMMAND[@]}" >> "${LOG_PATH}"
printf '\n' >> "${LOG_PATH}"
"${COMMAND[@]}" >> "${LOG_PATH}" 2>&1

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

echo "candidate complete: ${OUT_DIR}"
