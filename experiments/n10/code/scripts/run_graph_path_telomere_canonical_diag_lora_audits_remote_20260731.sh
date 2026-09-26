#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
    echo "usage: $0 GPU RUN_LABEL CHECKPOINT PHASE_JSON ARTIFACT OPERATOR_LABEL SAMPLE_SEED" >&2
    exit 2
fi

PHYSICAL_GPU="$1"
RUN_LABEL="$2"
CHECKPOINT="$3"
PHASE_SUMMARY="$4"
ARTIFACT="$5"
OPERATOR_LABEL="$6"
SAMPLE_SEED="$7"
SHARED_GPU="${SHARED_GPU:-0}"
DECLARED_PEAK_GIB="${DECLARED_PEAK_GIB:-4.0}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.05}"
RESERVE_GIB="${RESERVE_GIB:-16.0}"

CODE_DIR=/data/paperexperiment/LooPlus
PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
OUTPUT_ROOT=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/audits
LOG_ROOT=/data/paperexperiment/logs/graph_path_telomere_canonical_diag_lora_20260731
OUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"
RUN_LOG="${LOG_ROOT}/audit_${RUN_LABEL}.log"
MANIFEST="${OUT_DIR}/run_manifest.json"

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
PRELAUNCH_USED_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
PRELAUNCH_FREE_MIB="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
REQUIRED_FREE_MIB="$(awk -v peak="${DECLARED_PEAK_GIB}" -v reserve="${RESERVE_GIB}" 'BEGIN { printf "%d", (peak + reserve) * 1024 }')"
if [[ "${SHARED_GPU}" == "1" && "${PRELAUNCH_FREE_MIB}" -lt "${REQUIRED_FREE_MIB}" ]]; then
    echo "shared-GPU audit rejected: free=${PRELAUNCH_FREE_MIB}MiB required=${REQUIRED_FREE_MIB}MiB" >&2
    exit 75
fi
"${PYTHON_BIN}" - "${MANIFEST}" "${RUN_LABEL}" "${PHYSICAL_GPU}" "${ARTIFACT}" "${OPERATOR_LABEL}" "${PRELAUNCH_USED_MIB}" "${PRELAUNCH_FREE_MIB}" <<'PY'
import json
import sys
from pathlib import Path

path, label, gpu, artifact, operator, used, free = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "launching",
    "run_label": label,
    "physical_gpu": int(gpu),
    "prelaunch_used_mib": int(used),
    "prelaunch_free_mib": int(free),
    "artifact": artifact,
    "operator_label": operator,
    "audits": ["strict_unseen", "causal_path_and_gates", "schedule128", "boundary_dynamics", "spectrum"],
}, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

set +e
(
    set -e
    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --affine-artifact "$(dirname "${ARTIFACT}")/../../initializers/${RUN_LABEL}/unit_j_maps.pt" \
        --affine-label explicit_diag_rrr_r48 \
        --lora-artifacts "${ARTIFACT}" \
        --out-dir "${OUT_DIR}/strict_unseen" \
        --device cuda \
        --sample-per-partition 512 \
        --batch-size 32 \
        --continuation-loops 128 \
        --sample-seed "${SAMPLE_SEED}" \
        --cuda-memory-fraction "${CUDA_MEMORY_FRACTION}" \
        --training-graph-protocol canonical_artifact

    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_boundary_circuit \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --operator-artifact "${ARTIFACT}" \
        --operator-label "${OPERATOR_LABEL}" \
        --out-dir "${OUT_DIR}/circuit" \
        --device cuda \
        --batch-size 512 \
        --cycles 1 32 64 \
        --seed "$((SAMPLE_SEED + 1))" \
        --cuda-memory-fraction "${CUDA_MEMORY_FRACTION}"

    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_boundary_schedule \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --operator-artifact "${ARTIFACT}" \
        --operator-label "${OPERATOR_LABEL}" \
        --out-dir "${OUT_DIR}/schedule128" \
        --device cuda \
        --batch-size 128 \
        --batches 8 \
        --continuation-loops 128 \
        --seed "$((SAMPLE_SEED + 2))" \
        --cuda-memory-fraction "${CUDA_MEMORY_FRACTION}"

    "${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_boundary_dynamics \
        --checkpoint "${CHECKPOINT}" \
        --phase-summary "${PHASE_SUMMARY}" \
        --operator-artifact "${ARTIFACT}" \
        --operator-label "${OPERATOR_LABEL}" \
        --out-dir "${OUT_DIR}/dynamics" \
        --device cuda \
        --permutations-per-replica 64 \
        --replicas 2 \
        --continuation-loops 128 \
        --matched-period 8 \
        --commutator-cycles 1 32 64 96 128 \
        --sample-seed "$((SAMPLE_SEED + 3))" \
        --cuda-memory-fraction "${CUDA_MEMORY_FRACTION}"

    "${PYTHON_BIN}" scripts/analyze_diagonal_low_rank_spectrum.py \
        --artifact "${ARTIFACT}" \
        --label "${OPERATOR_LABEL}" \
        --out-dir "${OUT_DIR}/spectrum"
) > "${RUN_LOG}" 2>&1
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
exit "${STATUS}"
