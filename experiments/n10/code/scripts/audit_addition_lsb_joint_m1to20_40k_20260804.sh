#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL CONTROLLER CHECKPOINT" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
LABEL="$2"
CONTROLLER="$3"
CHECKPOINT="$4"

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
RUN_ROOT="${RUN_ROOT_OVERRIDE:-/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804/joint_m1to20_j_40k_20260804}"
OUT_ROOT="${RUN_ROOT}/audits/${LABEL}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/data/paperexperiment/logs/paper_length_telomere_20260731/joint_m1to20_j_40k_20260804}"
LOG_PATH="${LOG_ROOT}/${LABEL}_audit.log"
RESERVE_MIB=16384
DECLARED_PEAK_MIB=2048
REQUIRED_FREE_MIB=$((RESERVE_MIB + DECLARED_PEAK_MIB))
EVALUATION_SEED=1224001

if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
    echo "invalid physical GPU: ${PHYSICAL_GPU}" >&2
    exit 2
fi
if [[ ! -f "${CONTROLLER}" || ! -f "${CHECKPOINT}" ]]; then
    echo "missing controller or checkpoint" >&2
    exit 3
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

check_capacity() {
    local free_mib
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < REQUIRED_FREE_MIB )); then
        echo "GPU ${PHYSICAL_GPU} has ${free_mib} MiB free; ${REQUIRED_FREE_MIB} MiB required" >&2
        exit 75
    fi
}

run_eval() {
    local name="$1" batch_size="$2" batches="$3"
    shift 3
    local out_dir="${OUT_ROOT}/${name}"
    if [[ -f "${out_dir}/summary.json" ]] && grep -q '"status": "complete"' "${out_dir}/summary.json"; then
        echo "${name} already complete" >> "${LOG_PATH}"
        return
    fi
    check_capacity
    "${PYTHON_BIN}" -u scripts/evaluate_addition_controller_endpoint_accuracy.py \
        --checkpoint "${CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants raw full \
        --lengths "$@" \
        --batch-size "${batch_size}" \
        --batches "${batches}" \
        --seed "${EVALUATION_SEED}" \
        --device cuda \
        --out-dir "${out_dir}" >> "${LOG_PATH}" 2>&1
}

printf 'checkpoint=%q\ncontroller=%q\nphysical_gpu=%q\nevaluation_seed=%q\n' \
    "${CHECKPOINT}" "${CONTROLLER}" "${PHYSICAL_GPU}" "${EVALUATION_SEED}" > "${LOG_PATH}"

run_eval trained_l1to10_n1024 128 8 {1..10}
run_eval trained_l11to20_n1024 128 8 {11..20}
run_eval heldout_l21to30_n512 64 8 {21..30}

"${PYTHON_BIN}" - "${OUT_ROOT}/completion.json" "${LABEL}" "${PHYSICAL_GPU}" "${EVALUATION_SEED}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "complete",
    "label": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "evaluation_seed": int(sys.argv[4]),
    "controller_training_lengths": [1, 20],
    "structural_noop_lengths": [1],
    "heldout_lengths": [21, 30],
    "variants": ["raw", "full"],
    "finished_unix": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY

echo "audit complete: ${OUT_ROOT}"
