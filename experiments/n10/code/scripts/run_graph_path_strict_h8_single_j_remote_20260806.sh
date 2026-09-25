#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
ROOT=/data/wujiaju/graph_path_strict_h8_single_j_20260806
LOG_ROOT=/data/wujiaju/logs/graph_path_strict_h8_single_j_20260806
CHECKPOINT=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
INITIALIZER=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/initializers/final_seed0_ce_h64/unit_j_maps.pt
CONTROLLER_DIR="${ROOT}/controller"
AUDIT_DIR="${ROOT}/strict_unseen"
RAW_DIR="${ROOT}/raw_hidden_direction"
REPORT_DIR="${ROOT}/length_generalization"
LOG="${LOG_ROOT}/run.log"
MANIFEST="${ROOT}/run_manifest.json"

mkdir -p "${CONTROLLER_DIR}" "${AUDIT_DIR}" "${RAW_DIR}" "${REPORT_DIR}" "${LOG_ROOT}"
USED="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
FREE="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
"${PYTHON_BIN}" - "${MANIFEST}" "${GPU}" "${USED}" "${FREE}" <<'PY'
import json, sys
from pathlib import Path
path, gpu, used, free = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "running",
    "physical_gpu": int(gpu),
    "prelaunch_used_mib": int(used),
    "prelaunch_free_mib": int(free),
    "checkpoint": "/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt",
    "controller": "one shared loop-boundary J(h)=hD+(hA)B+b, rank 48",
    "training_support": "K in {1,2,4,6,8}; no training unroll exceeds 8",
    "loss": "successor CE only at every controlled continuation; hidden-state loss 0",
    "evaluation": "strictly unseen permutations, all starts, K=1..128",
}, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${CODE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR}"

set +e
{
"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_lora_j \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE}" \
    --reference-affine-artifact "${INITIALIZER}" \
    --reference-affine-label explicit_diag_rrr_r48 \
    --initialization affine_svd \
    --initial-affine-artifact "${INITIALIZER}" \
    --initial-affine-label explicit_diag_rrr_r48 \
    --require-affine-placement \
    --parameterization diagonal_low_rank \
    --diagonal-scale-init diagonal \
    --out-dir "${CONTROLLER_DIR}" \
    --device cuda \
    --placement loop_boundary \
    --ranks 48 \
    --initialization-seeds 211001 311001 411001 \
    --curriculum strict_h8 \
    --max-training-horizon 8 \
    --learning-rate-multiplier 30 \
    --scale-learning-rate-multiplier 0.1 \
    --state-loss-weight 0 \
    --evaluation-batch-size 128 \
    --evaluation-batches 8 \
    --evaluation-loops 128 \
    --evaluation-seed 212004 \
    --cuda-memory-fraction 0.06 \
    --physical-gpu "${GPU}" \
    --prelaunch-used-mib "${USED}" \
    --prelaunch-free-mib "${FREE}" \
    --declared-peak-gib 4 \
    --reserve-gib 16

"${PYTHON_BIN}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE}" \
    --affine-artifact "${INITIALIZER}" \
    --affine-label explicit_diag_rrr_r48 \
    --lora-artifacts "${CONTROLLER_DIR}/task_lora_j.pt" \
    --out-dir "${AUDIT_DIR}" \
    --device cuda \
    --sample-per-partition 512 \
    --batch-size 64 \
    --continuation-loops 128 \
    --sample-seed 20260806 \
    --cuda-memory-fraction 0.12 \
    --training-graph-protocol canonical_artifact

"${PYTHON_BIN}" -m reasoning_loop.graph_path_raw_hidden_direction \
    --checkpoint "${CHECKPOINT}" \
    --training-artifact "${CONTROLLER_DIR}/task_lora_j.pt" \
    --out-dir "${RAW_DIR}" \
    --device cuda \
    --loops 64 \
    --permutations 64 \
    --sample-seed 20260806 \
    --cuda-memory-fraction 0.08

"${PYTHON_BIN}" -m reasoning_loop.graph_path_strict_h8_report \
    --training-summary "${CONTROLLER_DIR}/summary.json" \
    --strict-unseen-summary "${AUDIT_DIR}/summary.json" \
    --out-dir "${REPORT_DIR}"
} >"${LOG}" 2>&1
STATUS=$?
set -e

"${PYTHON_BIN}" - "${MANIFEST}" "${STATUS}" <<'PY'
import json, sys
from pathlib import Path
path, code = sys.argv[1:]
p = Path(path)
payload = json.loads(p.read_text())
payload["exit_code"] = int(code)
payload["status"] = "complete" if int(code) == 0 else "failed"
p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
exit "${STATUS}"
