#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 BACKBONE_SEED" >&2
    exit 2
fi

BACKBONE_SEED="$1"
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${CODE_DIR}/.venv/bin/python"
RUN_ROOT="${CODE_DIR}/results/paper_length_telomere_20260731/local_mps_released64_formal"
LABEL="parity_adaptive_step_released64_seed${BACKBONE_SEED}"
BACKBONE_DIR="${RUN_ROOT}/backbones/${LABEL}"
DIAGNOSIS_DIR="${RUN_ROOT}/diagnosis/${LABEL}"
CONTROLLER_SEED=211001
CONTROLLER_CURRICULUM=extension
CONTROLLER_LABEL="${LABEL}_rank48_${CONTROLLER_CURRICULUM}_seed${CONTROLLER_SEED}"
CONTROLLER_DIR="${RUN_ROOT}/controllers/${CONTROLLER_LABEL}"
AUDIT_DIR="${RUN_ROOT}/audits/${CONTROLLER_LABEL}"

mkdir -p "${BACKBONE_DIR}" "${DIAGNOSIS_DIR}" "${CONTROLLER_DIR}" "${AUDIT_DIR}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
cd "${CODE_DIR}"

summary_is_complete() {
    local summary="$1"
    [[ -f "${summary}" ]] && grep -q '"status": "complete"' "${summary}"
}

run_with_retries() {
    local stage="$1"
    local summary="$2"
    local maximum_attempts="$3"
    shift 3
    local attempt=0
    local status=0
    while ! summary_is_complete "${summary}"; do
        attempt=$((attempt + 1))
        echo "$(date '+%Y-%m-%dT%H:%M:%S%z') local_mps stage=${stage} seed=${BACKBONE_SEED} attempt=${attempt}"
        set +e
        "$@"
        status=$?
        set -e
        if [[ "${status}" -ne 0 ]]; then
            echo "$(date '+%Y-%m-%dT%H:%M:%S%z') local_mps stage=${stage} seed=${BACKBONE_SEED} attempt=${attempt} exit=${status}"
            if [[ "${attempt}" -ge "${maximum_attempts}" ]]; then
                echo "stage ${stage} exhausted ${maximum_attempts} attempts" >&2
                return "${status}"
            fi
            sleep 30
        fi
    done
}

run_backbone() {
    local resume_checkpoint=""
    local command=(
        "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere backbone
        --task parity --supervision adaptive_step --steps 100001
        --n-heads 64 --no-amp --device mps --seed "${BACKBONE_SEED}"
        --log-every 100 --checkpoint-every 10000
    )
    resume_checkpoint="$(find "${BACKBONE_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.pt' | sort | tail -n 1)"
    if [[ -n "${resume_checkpoint}" ]]; then
        command+=(--resume "${resume_checkpoint}")
    fi
    command+=(--out-dir "${BACKBONE_DIR}")
    "${command[@]}"
}

run_diagnosis() {
    "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere diagnose \
        --checkpoint "${BACKBONE_DIR}/final.pt" \
        --batch-size 128 --batches 32 --lengths 20 40 50 75 100 \
        --maximum-step 132 --extension-gate-length 40 \
        --device mps --out-dir "${DIAGNOSIS_DIR}"
}

run_controller() {
    "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere controller \
        --checkpoint "${BACKBONE_DIR}/final.pt" \
        --diagnosis "${DIAGNOSIS_DIR}/summary.json" --rank 48 \
        --seed "${CONTROLLER_SEED}" \
        --controller-curriculum "${CONTROLLER_CURRICULUM}" \
        --force --device mps --out-dir "${CONTROLLER_DIR}"
}

run_audit() {
    "${PYTHON_BIN}" -m reasoning_loop.paper_length_telomere audit \
        --checkpoint "${BACKBONE_DIR}/final.pt" \
        --controller "${CONTROLLER_DIR}/controller.pt" \
        --batch-size 128 --batches 32 --lengths 20 40 50 75 84 100 \
        --maximum-step 132 --seed 261001 \
        --device mps --out-dir "${AUDIT_DIR}"
}

run_with_retries backbone "${BACKBONE_DIR}/summary.json" 5 run_backbone
run_with_retries diagnosis "${DIAGNOSIS_DIR}/summary.json" 3 run_diagnosis
run_with_retries controller "${CONTROLLER_DIR}/summary.json" 3 run_controller
run_with_retries audit "${AUDIT_DIR}/summary.json" 3 run_audit

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') local MPS released64 telomere pipeline complete seed=${BACKBONE_SEED}"
