#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 MODE TASK PHYSICAL_GPU" >&2
    exit 2
fi

MODE="$1"
TASK="$2"
PHYSICAL_GPU="$3"
case "${MODE}" in
    benchmark|formal) ;;
    *) echo "MODE must be benchmark or formal" >&2; exit 2 ;;
esac
case "${TASK}" in
    copy)
        SLOT=0
        BENCHMARK_DECLARED_PEAK_MIB=3072
        BENCHMARK_MEMORY_FRACTION=0.03000
        ;;
    addition)
        SLOT=1
        BENCHMARK_DECLARED_PEAK_MIB=6144
        BENCHMARK_MEMORY_FRACTION=0.07000
        ;;
    sum_reverse)
        SLOT=2
        BENCHMARK_DECLARED_PEAK_MIB=4096
        BENCHMARK_MEMORY_FRACTION=0.04500
        ;;
    *) echo "unsupported task: ${TASK}" >&2; exit 2 ;;
esac
if ! [[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]]; then
    echo "PHYSICAL_GPU must be in [0,7]" >&2
    exit 2
fi

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
RUNNER=/data/paperexperiment/LooPlus/scripts/run_paper_length_telomere_remote_20260731.sh
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
LOCK_ROOT="${RUN_ROOT}/locks"
LABEL="${TASK}_adaptive_step_official_seed0"
BENCHMARK_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/official_benchmark_fp32.json"
BENCHMARK_SUMMARY="${RUN_ROOT}/backbones/${LABEL}_benchmark_fp32/summary.json"
RESERVE_MIB=16384
mkdir -p "${LOCK_ROOT}"

manifest_is_complete() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && grep -q '"status": "complete"' "${manifest}"
}

run_shared() {
    local action="$1"
    local seed="$2"
    local declared_peak_mib="$3"
    local memory_fraction="$4"
    local manifest="$5"
    if manifest_is_complete "${manifest}"; then
        echo "$(date -Is) shared stage already complete action=${action} task=${TASK} seed=${seed}"
        return 0
    fi
    (
        flock -x 8
        flock -x 9
        while true; do
            local free_mib
            free_mib="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
            if [[ "${free_mib}" -ge $((RESERVE_MIB + declared_peak_mib)) ]]; then
                break
            fi
            echo "$(date -Is) waiting shared memory action=${action} task=${TASK} seed=${seed} gpu=${PHYSICAL_GPU} free=${free_mib} required=$((RESERVE_MIB + declared_peak_mib))"
            sleep 30
        done
        echo "$(date -Is) launching shared action=${action} task=${TASK} seed=${seed} gpu=${PHYSICAL_GPU} declared_peak_mib=${declared_peak_mib} reserve_mib=${RESERVE_MIB}"
        env \
            BASELINE_VARIANT=official \
            ALLOW_SHARED_GPU=1 \
            DECLARED_PEAK_MIB="${declared_peak_mib}" \
            RESERVE_MIB="${RESERVE_MIB}" \
            PAPER_CUDA_MEMORY_FRACTION="${memory_fraction}" \
            bash "${RUNNER}" "${action}" "${TASK}" adaptive_step "${PHYSICAL_GPU}" "${seed}"
    ) 8>"${LOCK_ROOT}/paper_job_slot${SLOT}.lock" 9>"${LOCK_ROOT}/gpu${PHYSICAL_GPU}.lock"
}

if [[ "${MODE}" == benchmark ]]; then
    # Task-specific allocator caps bound the first co-located measurement.  If
    # a task needs more, it fails locally instead of consuming the 16 GiB
    # reserve owned by the pre-existing process.
    run_shared official_benchmark_fp32 0 \
        "${BENCHMARK_DECLARED_PEAK_MIB}" \
        "${BENCHMARK_MEMORY_FRACTION}" \
        "${BENCHMARK_MANIFEST}"
    exit 0
fi

if ! manifest_is_complete "${BENCHMARK_MANIFEST}"; then
    echo "formal shared run requires a completed matching FP32 benchmark" >&2
    exit 1
fi

DECLARED_PEAK_MIB="$("${PYTHON_BIN}" - "${BENCHMARK_SUMMARY}" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
peak_gib = float(payload["peak_cuda_memory_reserved_gib"])
# Include CUDA-context and allocator headroom beyond max_memory_reserved.
print(max(1, math.ceil(peak_gib * 1024) + 1024))
PY
)"
MEMORY_FRACTION="$("${PYTHON_BIN}" - "${DECLARED_PEAK_MIB}" <<'PY'
import sys

declared = int(sys.argv[1])
print(f"{max(0.01875, (declared + 128) / 81920):.8f}")
PY
)"

for SEED in 0 1 2; do
    FORMAL_LABEL="${TASK}_adaptive_step_official_seed${SEED}"
    FORMAL_MANIFEST="${RUN_ROOT}/manifests/${FORMAL_LABEL}/official_formal.json"
    run_shared official_formal "${SEED}" "${DECLARED_PEAK_MIB}" "${MEMORY_FRACTION}" "${FORMAL_MANIFEST}"
done

echo "$(date -Is) shared official baselines complete task=${TASK} seeds=0,1,2 gpu=${PHYSICAL_GPU}"
