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
OUT_ROOT="${RUN_ROOT}/copy4_controls_20260804"
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731/copy4_controls_20260804
DECLARED_PEAK_MIB=2048
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))
LENGTHS=(1 5 10 15 19 20 25 30 35 40 45 50 60)

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

check_capacity() {
    local phase="$1" free_mib data_free_gib
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    data_free_gib="$(df -BG --output=avail /data | tail -n 1 | tr -dc '0-9')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "${phase}: GPU ${PHYSICAL_GPU} has ${free_mib}MiB free; ${REQUIRED_FREE_MIB}MiB required" >&2
        exit 75
    fi
    if [[ "${data_free_gib}" -lt 50 ]]; then
        echo "${phase}: /data has only ${data_free_gib}GiB free" >&2
        exit 75
    fi
}

run_mode() {
    local seed="$1" mode="$2" executor_off="$3"
    local controller_seed baseline_label controller_label backbone controller out_dir log_path
    controller_seed=$((211001 + seed))
    baseline_label="copy4_adaptive_step_official_seed${seed}"
    controller_label="${baseline_label}_rank48_identitywarmup_logical1to20_fullrange_anchor1_postfinal_wsd_lr1em4_warmup2048_stable2816_5k_cseed${controller_seed}"
    backbone="${RUN_ROOT}/backbones/${baseline_label}/final.pt"
    controller="${RUN_ROOT}/controllers/${controller_label}/controller.pt"
    out_dir="${OUT_ROOT}/seed${seed}/${mode}"
    log_path="${LOG_ROOT}/seed${seed}_${mode}.log"

    if [[ ! -f "${backbone}" ]] || [[ ! -f "${controller}" ]]; then
        echo "seed ${seed}: missing matched backbone or controller" >&2
        exit 3
    fi
    if [[ -f "${out_dir}/summary.json" ]] && grep -q '"status": "complete"' "${out_dir}/summary.json"; then
        echo "seed=${seed} mode=${mode} already complete"
        return
    fi
    check_capacity "seed${seed}_${mode}"
    mkdir -p "${out_dir}"
    command=(
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py
        --task copy4
        --checkpoint "${backbone}"
        --controller "${controller}"
        --lengths "${LENGTHS[@]}"
        --examples 512
        --max-batch-size 32
        --token-budget 8192
        --seed 381001
        --device cuda
        --post-final-controller
        --controller-mode "${mode%_executor_off}"
        --out-dir "${out_dir}"
    )
    if [[ "${executor_off}" == "true" ]]; then
        command+=(--executor-off-after-anchor)
    fi
    printf 'launch_command=' > "${log_path}"
    printf '%q ' "${command[@]}" >> "${log_path}"
    printf '\n' >> "${log_path}"
    "${command[@]}" >> "${log_path}" 2>&1
}

for seed in 0 1 2; do
    run_mode "${seed}" full false
    run_mode "${seed}" no_AB false
    run_mode "${seed}" identity_D false
    run_mode "${seed}" full_executor_off true
done

"${PYTHON_BIN}" - "${OUT_ROOT}/completion.json" "${PHYSICAL_GPU}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "complete",
    "finished_unix": time.time(),
    "physical_gpu": int(sys.argv[2]),
    "backbone_seeds": [0, 1, 2],
    "evaluation_seed": 381001,
    "examples_per_length": 512,
    "lengths": [1, 5, 10, 15, 19, 20, 25, 30, 35, 40, 45, 50, 60],
    "modes": ["raw", "full", "no_AB", "identity_D", "full_executor_off"],
    "post_final_j": True,
}, indent=2, sort_keys=True) + "\n")
PY
