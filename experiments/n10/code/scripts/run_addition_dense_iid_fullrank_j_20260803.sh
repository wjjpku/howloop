#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]] || [[ "$#" -gt 2 ]] || ! [[ "$1" =~ ^[0-7]$ ]]; then
    echo "usage: $0 PHYSICAL_GPU [standard|fixed_n10_t11]" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
EXPERIMENT_VARIANT="${2:-standard}"
CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
LOG_ROOT=/data/paperexperiment/logs/paper_length_telomere_20260731
case "${EXPERIMENT_VARIANT}" in
    standard)
        RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/dense_iid_fullrank_addition_20260803
        BACKBONE=/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_adaptive_step_official_seed0/final.pt
        EXPECTED_BACKBONE_STEP=100001
        EXPECTED_FIXED_LOGICAL_LENGTH=none
        BACKBONE_DESCRIPTION="official Addition seed-0, update 100001, adaptive-step CE, n=1..19"
        TRAIN_LOGICAL_MIN=20
        TRAIN_LOGICAL_MAX=40
        EVAL_LENGTHS=(19 20 24 25 30 35 38 40 50 60)
        ;;
    fixed_n10_t11)
        RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731/dense_iid_fullrank_addition_fixed_n10_t11_20260803
        BACKBONE=/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_fixed_n10_t11_official_seed0/checkpoint_040000.pt
        EXPECTED_BACKBONE_STEP=40000
        EXPECTED_FIXED_LOGICAL_LENGTH=10
        BACKBONE_DESCRIPTION="official Addition seed-0, update 40000, fixed n=10 with T(n)=11"
        TRAIN_LOGICAL_MIN=1
        TRAIN_LOGICAL_MAX=10
        EVAL_LENGTHS=(1 5 10 11 12 15 20 25 30 40)
        ;;
    *)
        echo "variant must be standard or fixed_n10_t11" >&2
        exit 2
        ;;
esac
CONTROLLER_DIR="${RUN_ROOT}/controller"
CONTROLLER="${CONTROLLER_DIR}/controller.pt"
INITIAL_CONTROLLER="${CONTROLLER_DIR}/checkpoints/controller_000000.pt"
INITIAL_EVAL_DIR="${RUN_ROOT}/eval_initial_iid"
FINAL_EVAL_DIR="${RUN_ROOT}/eval_trained"
MANIFEST_PATH="${RUN_ROOT}/pipeline_manifest.json"
HEARTBEAT_PATH="${RUN_ROOT}/heartbeat.json"
LOG_PATH="${LOG_ROOT}/addition_dense_iid_fullrank_j_${EXPERIMENT_VARIANT}_20260803.log"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384
REQUIRED_FREE_MIB=$((DECLARED_PEAK_MIB + RESERVE_MIB))

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
UTILIZATION="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
ACTIVE_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ "${FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "GPU ${PHYSICAL_GPU}: free=${FREE_MIB}MiB, required=${REQUIRED_FREE_MIB}MiB" >&2
    exit 75
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.085
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

"${PYTHON_BIN}" - "${BACKBONE}" "${EXPECTED_BACKBONE_STEP}" \
    "${EXPECTED_FIXED_LOGICAL_LENGTH}" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert payload["kind"] == "paper_length_telomere_backbone"
assert payload["task"]["name"] == "addition"
assert payload["step"] == int(sys.argv[2])
assert payload["supervision"] == "adaptive_step"
assert payload["model"]["d_model"] == 256
expected_fixed = None if sys.argv[3] == "none" else int(sys.argv[3])
assert payload.get("training_fixed_logical_length") == expected_fixed
PY

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${USED_MIB}" \
    "${FREE_MIB}" "${UTILIZATION}" "${ACTIVE_PIDS}" \
    "${EXPERIMENT_VARIANT}" "${BACKBONE_DESCRIPTION}" \
    "${TRAIN_LOGICAL_MIN}" "${TRAIN_LOGICAL_MAX}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "launching",
    "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]),
    "prelaunch_used_mib": int(sys.argv[3]),
    "prelaunch_free_mib": int(sys.argv[4]),
    "prelaunch_utilization_percent": int(sys.argv[5]),
    "preexisting_pids": sys.argv[6].split(),
    "experiment_variant": sys.argv[7],
    "declared_matching_peak_mib": 6144,
    "reserve_mib": 16384,
    "backbone": sys.argv[8],
    "controller": "full-rank J(h)=hW+b; 256*257=65792 parameters",
    "initialization": {
        "diagonal": 1.0,
        "off_diagonal_distribution": "iid Normal(0, (1/225)^2)",
        "bias_distribution": "iid Normal(0, 0.01^2)",
        "expected_off_diagonal_residual_rms": 0.0709720863229836,
        "expected_bias_rms": 0.01,
    },
    "training": {
        "logical_lengths": [int(sys.argv[9]), int(sys.argv[10])],
        "anchor_step": 1,
        "post_final_j": True,
        "supervision": "full-answer final CE only",
        "optimizer_updates": 5376,
        "lr_schedule": "WSD: warmup 2048, stable 2816, decay 512",
        "peak_learning_rate": 0.0001,
        "gradient_clip": 1.0,
    },
}, indent=2, sort_keys=True) + "\n")
PY

update_manifest() {
    local status="$1" phase="$2" child_pid="$3" active_log="$4"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${phase}" \
        "${child_pid}" "${active_log}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": sys.argv[2],
    "active_phase": sys.argv[3] or None,
    "pid": int(sys.argv[4]) if sys.argv[4] else None,
    "active_log": sys.argv[5] or None,
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_monitored() {
    local phase="$1"
    shift
    local child_pid free_mib used_mib utilization status
    free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -lt "${REQUIRED_FREE_MIB}" ]]; then
        echo "GPU ${PHYSICAL_GPU}: free=${free_mib}MiB before ${phase}" >&2
        return 75
    fi
    printf 'phase=%s launch_command=' "${phase}" >> "${LOG_PATH}"
    printf '%q ' "$@" >> "${LOG_PATH}"
    printf '\n' >> "${LOG_PATH}"
    "$@" >> "${LOG_PATH}" 2>&1 &
    child_pid=$!
    update_manifest running "${phase}" "${child_pid}" "${LOG_PATH}"
    while kill -0 "${child_pid}" 2>/dev/null; do
        used_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        utilization="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        "${PYTHON_BIN}" - "${HEARTBEAT_PATH}" "${phase}" "${child_pid}" \
            "${PHYSICAL_GPU}" "${used_mib}" "${free_mib}" "${utilization}" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "unix": time.time(),
    "active_phase": sys.argv[2],
    "child_pid": int(sys.argv[3]),
    "physical_gpu": int(sys.argv[4]),
    "used_mib": int(sys.argv[5]),
    "free_mib": int(sys.argv[6]),
    "utilization_percent": int(sys.argv[7]),
}, indent=2, sort_keys=True) + "\n")
PY
        if [[ "${free_mib}" -lt "${RESERVE_MIB}" ]]; then
            echo "reserve guard: free=${free_mib}MiB at ${phase}" >> "${LOG_PATH}"
            kill -TERM "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" || true
            update_manifest failed "${phase}" "" "${LOG_PATH}"
            return 76
        fi
        for _ in {1..15}; do
            sleep 2
            if ! kill -0 "${child_pid}" 2>/dev/null; then break; fi
        done
    done
    status=0
    wait "${child_pid}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
        update_manifest failed "${phase}" "" "${LOG_PATH}"
    fi
    return "${status}"
}

if [[ ! -f "${CONTROLLER_DIR}/summary.json" ]] || \
   ! grep -q '"status": "complete"' "${CONTROLLER_DIR}/summary.json"; then
    run_monitored train_dense_iid_fullrank_5376 \
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
        --checkpoint "${BACKBONE}" \
        --controller-parameterization dense_affine \
        --controller-initialization dense_iid \
        --dense-iid-off-diagonal-std 0.0044444444444444444 \
        --dense-iid-bias-rms 0.01 \
        --dense-stage-count 0 \
        --seed 211001 \
        --controller-curriculum logical_range \
        --controller-logical-min-length "${TRAIN_LOGICAL_MIN}" \
        --controller-logical-max-length "${TRAIN_LOGICAL_MAX}" \
        --controller-training-profile identity_long_warmup \
        --controller-anchor-step 1 \
        --controller-warmup-updates 2048 \
        --controller-final-lr-ratio 0.1 \
        --controller-lr-schedule wsd \
        --controller-stable-updates 2816 \
        --controller-post-final-j \
        --controller-supervision full_answer \
        --controller-checkpoint-every 256 \
        --grad-clip 1.0 \
        --learning-rate-multiplier 5.0 \
        --stage-round-multiplier 3 \
        --force \
        --device cuda \
        --out-dir "${CONTROLLER_DIR}"
fi

[[ -f "${CONTROLLER}" ]] || { echo "missing ${CONTROLLER}" >&2; exit 4; }
[[ -f "${INITIAL_CONTROLLER}" ]] || { echo "missing ${INITIAL_CONTROLLER}" >&2; exit 5; }
if [[ ! -f "${INITIAL_EVAL_DIR}/summary.json" ]]; then
    run_monitored eval_initial_iid \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${BACKBONE}" \
        --controller "${INITIAL_CONTROLLER}" \
        --lengths "${EVAL_LENGTHS[@]}" --examples 512 \
        --max-batch-size 32 --token-budget 8192 --seed 261001 \
        --device cuda --out-dir "${INITIAL_EVAL_DIR}"
fi

if [[ ! -f "${FINAL_EVAL_DIR}/summary.json" ]]; then
    run_monitored eval_trained \
        "${PYTHON_BIN}" scripts/evaluate_parity_far_horizon.py \
        --task addition --checkpoint "${BACKBONE}" \
        --controller "${CONTROLLER}" \
        --lengths "${EVAL_LENGTHS[@]}" --examples 512 \
        --max-batch-size 32 --token-budget 8192 --seed 261001 \
        --device cuda --out-dir "${FINAL_EVAL_DIR}"
fi

update_manifest complete "" "" ""
