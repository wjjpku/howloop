#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CODE=/data/paperexperiment/LooPlus
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
ROOT=/data/paperexperiment/graph_path_strict_dense_h8_continue_20260806
LOG_ROOT=/data/paperexperiment/logs/graph_path_strict_dense_h8_continue_20260806
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
AFFINE=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/initializers/final_seed0_ce_h64/unit_j_maps.pt
BEFORE=/data/paperexperiment/graph_path_strict_h8_single_j_20260806/controller/task_lora_j.pt
CONTROLLER="${ROOT}/controller"
AUDIT="${ROOT}/matched_strict_unseen"
REPORT="${ROOT}/comparison"
LOG="${LOG_ROOT}/run.log"
MANIFEST="${ROOT}/run_manifest.json"
mkdir -p "${CONTROLLER}" "${AUDIT}" "${REPORT}" "${LOG_ROOT}"

USED="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
FREE="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
"${PYTHON}" - "${MANIFEST}" "${GPU}" "${USED}" "${FREE}" <<'PY'
import json, sys
from pathlib import Path
path,gpu,used,free=sys.argv[1:]
Path(path).write_text(json.dumps({
  "status":"running",
  "physical_gpu":int(gpu),
  "prelaunch_used_mib":int(used),
  "prelaunch_free_mib":int(free),
  "operation":"continue three existing rank48 J controllers",
  "dense_training_support":[1,2,3,4,5,6,7,8],
  "sampling":"each horizon exactly once per round and per controller",
  "rounds_per_controller":64,
  "loss":"successor CE only; no hidden-state loss",
  "comparison":"before and after on identical permutations unseen by all after-training streams",
},indent=2,sort_keys=True)+"\n")
PY

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${CODE}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE}"
set +e
{
"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_dense_continue \
  --checkpoint "${CHECKPOINT}" \
  --phase-summary "${PHASE}" \
  --source-artifact "${BEFORE}" \
  --out-dir "${CONTROLLER}" \
  --device cuda \
  --rounds 64 \
  --batch-size 64 \
  --learning-rate 3e-5 \
  --diagonal-learning-rate 3e-6 \
  --data-seed 189003 \
  --evaluation-batch-size 128 \
  --evaluation-batches 8 \
  --evaluation-loops 128 \
  --evaluation-seed 212004 \
  --cuda-memory-fraction 0.06 \
  --physical-gpu "${GPU}" \
  --prelaunch-used-mib "${USED}" \
  --prelaunch-free-mib "${FREE}"

"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_matched_audit \
  --checkpoint "${CHECKPOINT}" \
  --phase-summary "${PHASE}" \
  --before-artifact "${BEFORE}" \
  --after-artifact "${CONTROLLER}/task_lora_j.pt" \
  --out-dir "${AUDIT}" \
  --device cuda \
  --permutations 512 \
  --batch-size 64 \
  --loops 128 \
  --sample-seed 20260806 \
  --cuda-memory-fraction 0.16

"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_dense_compare \
  --audit "${AUDIT}/summary.json" \
  --out-dir "${REPORT}"
} >"${LOG}" 2>&1
STATUS=$?
set -e
"${PYTHON}" - "${MANIFEST}" "${STATUS}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); code=int(sys.argv[2])
d["exit_code"]=code; d["status"]="complete" if code==0 else "failed"
p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY
exit "${STATUS}"
