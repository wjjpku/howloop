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
SOURCE_LABEL=addition_adaptive_step_official_seed0_rank48_identitywarmup_logical1to20_fullrange_anchor1_postfinal_wsd_lr1em4_warmup2048_stable2816_5k_seed211001
SOURCE_CONTROLLER="${RUN_ROOT}/controllers/${SOURCE_LABEL}/controller.pt"
PERIODIC_LABEL=addition_adaptive_step_official_seed0_rank48_logical1to20_fullrange_anchor1_postfinal_continue5376to50000_mix1_10_15_20_wsd5k_35k_4624_lr1em4_seed311001
PERIODIC_CONTROLLER="${RUN_ROOT}/controllers/${PERIODIC_LABEL}/controller.pt"
ROBUST_LABEL=addition_adaptive_step_official_seed0_rank48_robust_agepath_logical1to20_continue5376to50000_wsd5k_35k_4624_lr1em4_seed411001
OUT_DIR="${RUN_ROOT}/controllers/${ROBUST_LABEL}"
AUDIT_ROOT="${RUN_ROOT}/robust_age_path_20260803"
PIPELINE_LOG="${LOG_ROOT}/${ROBUST_LABEL}_pipeline.log"
MANIFEST_PATH="${OUT_DIR}/pipeline_manifest.json"
DECLARED_PEAK_MIB=6144
RESERVE_MIB=16384

mkdir -p "${OUT_DIR}" "${AUDIT_ROOT}" "${LOG_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

BACKBONE="${RUN_ROOT}/backbones/addition_adaptive_step_official_seed0/final.pt"
for required in "${BACKBONE}" "${SOURCE_CONTROLLER}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required artifact: ${required}" >&2
        exit 2
    fi
done

free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
required_free_mib=$((DECLARED_PEAK_MIB + RESERVE_MIB))
if [[ "${free_mib}" -lt "${required_free_mib}" ]]; then
    echo "GPU ${PHYSICAL_GPU} has ${free_mib}MiB free; ${required_free_mib}MiB required" >&2
    exit 75
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PAPER_CUDA_MEMORY_FRACTION=0.075
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PHYSICAL_GPU}" "${BACKBONE}" \
    "${SOURCE_CONTROLLER}" "${PERIODIC_CONTROLLER}" "${OUT_DIR}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "running",
    "created_unix": time.time(),
    "physical_gpu": int(sys.argv[2]),
    "backbone": sys.argv[3],
    "source_controller": sys.argv[4],
    "compute_matched_periodic_controller": sys.argv[5],
    "output_dir": sys.argv[6],
    "task": "addition",
    "controller": "rank-48 diagonal + AB + bias",
    "strict_training_logical_range": [1, 20],
    "target_total_updates": 50000,
    "loss": "final answer-region CE only",
    "path_semantics": "balanced Dyck-like words with equal F/J counts",
    "maximum_age_curriculum": [1, 2, 4, 8, 20],
    "heldout_path_family": "all F followed by all J",
    "declared_peak_mib": 6144,
    "reserve_mib": 16384,
    "completed_phases": [],
}, indent=2, sort_keys=True) + "\n")
PY

mark_phase() {
    local status="$1"
    local phase="$2"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${phase}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
phase = sys.argv[3]
completed = list(payload.get("completed_phases", []))
if sys.argv[2] == "complete_phase" and phase not in completed:
    completed.append(phase)
payload.update({
    "status": "running" if sys.argv[2] == "complete_phase" else sys.argv[2],
    "active_phase": None if sys.argv[2] == "complete_phase" else phase,
    "completed_phases": completed,
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

mark_phase running train_robust_age_path
RESUME_ARGS=()
if [[ -f "${OUT_DIR}/latest.pt" ]]; then
    RESUME_ARGS=(--resume "${OUT_DIR}/latest.pt")
fi
"${PYTHON_BIN}" -m reasoning_loop.robust_age_path_controller train \
    --checkpoint "${BACKBONE}" \
    --source-controller "${SOURCE_CONTROLLER}" \
    --target-total-update 50000 \
    --batch-size 32 \
    --retention-maximum 10 \
    --transition-maximum 15 \
    --boundary-maximum 20 \
    --retention-probability 0.20 \
    --transition-probability 0.30 \
    --boundary-probability 0.50 \
    --peak-learning-rate 1e-4 \
    --diagonal-lr-multiplier 0.1 \
    --warmup-updates 5000 \
    --stable-updates 35000 \
    --final-learning-rate-ratio 0.1 \
    --grad-clip 1.0 \
    --seed 411001 \
    --eval-seed 461001 \
    --eval-lengths 10 15 19 20 \
    --eval-batch-size 64 \
    --eval-batches 2 \
    --eval-every 1000 \
    --log-every 100 \
    --checkpoint-every 5000 \
    --device cuda \
    --out-dir "${OUT_DIR}" \
    "${RESUME_ARGS[@]}"
mark_phase complete_phase train_robust_age_path

LENGTHS=(1 5 10 15 19 20 21 25 30 35 40 45 50 60 75 100)
mark_phase running audit_robust
"${PYTHON_BIN}" -m reasoning_loop.robust_age_path_controller audit \
    --checkpoint "${BACKBONE}" \
    --controller "${OUT_DIR}/controller.pt" \
    --lengths "${LENGTHS[@]}" \
    --batch-size 64 \
    --batches 4 \
    --seed 471001 \
    --path-seed 481001 \
    --device cuda \
    --out-dir "${AUDIT_ROOT}/${ROBUST_LABEL}"
mark_phase complete_phase audit_robust

if [[ -f "${PERIODIC_CONTROLLER}" ]]; then
    mark_phase running audit_compute_matched_periodic
    "${PYTHON_BIN}" -m reasoning_loop.robust_age_path_controller audit \
        --checkpoint "${BACKBONE}" \
        --controller "${PERIODIC_CONTROLLER}" \
        --lengths "${LENGTHS[@]}" \
        --batch-size 64 \
        --batches 4 \
        --seed 471001 \
        --path-seed 481001 \
        --device cuda \
        --out-dir "${AUDIT_ROOT}/${PERIODIC_LABEL}_path_audit"
    mark_phase complete_phase audit_compute_matched_periodic
else
    echo "periodic 50k controller not ready; matched audit deferred"
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json, pathlib, time, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "active_phase": None,
    "finished_unix": time.time(),
    "updated_unix": time.time(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
