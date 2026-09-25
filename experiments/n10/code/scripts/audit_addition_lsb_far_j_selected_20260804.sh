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

CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804/far_j_search_20260804
OUT_ROOT="${RUN_ROOT}/audits/${LABEL}"
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731/far_j_search_20260804
LOG_PATH="${LOG_ROOT}/${LABEL}_audit.log"
RESERVE_MIB=16384
DECLARED_PEAK_MIB=2048
REQUIRED_FREE_MIB=$((RESERVE_MIB + DECLARED_PEAK_MIB))

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
    local name="$1" batch_size="$2" batches="$3" variants="$4"
    shift 4
    local out_dir="${OUT_ROOT}/${name}"
    if [[ -f "${out_dir}/summary.json" ]] && grep -q '"status": "complete"' "${out_dir}/summary.json"; then
        echo "${name} already complete" >> "${LOG_PATH}"
        return
    fi
    check_capacity
    # The variants and lengths originate from fixed arrays below, not user text.
    read -r -a variant_array <<< "${variants}"
    "${PYTHON_BIN}" -u scripts/evaluate_addition_controller_endpoint_accuracy.py \
        --checkpoint "${CHECKPOINT}" \
        --controller "${CONTROLLER}" \
        --variants "${variant_array[@]}" \
        --lengths "$@" \
        --batch-size "${batch_size}" \
        --batches "${batches}" \
        --seed 784001 \
        --device cuda \
        --out-dir "${out_dir}" >> "${LOG_PATH}" 2>&1
}

printf 'checkpoint=%q\ncontroller=%q\nphysical_gpu=%q\n' \
    "${CHECKPOINT}" "${CONTROLLER}" "${PHYSICAL_GPU}" > "${LOG_PATH}"

run_eval id_l1to10_n1024 128 8 "raw full" {1..10}
run_eval exposed_l11to20_n1024 128 8 "raw full" {11..20}
run_eval unopened_l21to30_n512 64 8 "raw full" {21..30}
PARAMETERIZATION="$("${PYTHON_BIN}" - "${CONTROLLER}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(payload.get("controller_parameterization", "diagonal_low_rank"))
PY
)"
if [[ "${PARAMETERIZATION}" == "dense_affine" ]]; then
    COMPONENT_CONTROL=no_offdiag
else
    COMPONENT_CONTROL=no_AB
fi
run_eval controls_n512 64 8 "raw full ${COMPONENT_CONTROL} identity_D full_executor_off" \
    10 12 14 16 18 20 22 25 30

"${PYTHON_BIN}" - "${OUT_ROOT}/completion.json" "${LABEL}" "${PHYSICAL_GPU}" "${PARAMETERIZATION}" "${COMPONENT_CONTROL}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "complete",
    "label": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "controller_parameterization": sys.argv[4],
    "evaluation_seed": 784001,
    "id_lengths": [1, 10],
    "controller_exposed_lengths": [11, 20],
    "unopened_test_lengths": [21, 30],
    "controls": ["raw", "full", sys.argv[5], "identity_D", "full_executor_off"],
    "finished_unix": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY

echo "audit complete: ${OUT_ROOT}"
