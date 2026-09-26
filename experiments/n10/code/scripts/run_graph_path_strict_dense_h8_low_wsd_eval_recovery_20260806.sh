#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
CODE=/data/paperexperiment/LooPlus
PYTHON=/data/paperexperiment/.venvs/loopreasoner/bin/python
ROOT=/data/paperexperiment/graph_path_strict_dense_h8_low_wsd_20260806
LOG_ROOT=/data/paperexperiment/logs/graph_path_strict_dense_h8_low_wsd_20260806
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
BEFORE=/data/paperexperiment/graph_path_strict_dense_h8_continue_20260806/controller/task_lora_j.pt
AFTER="${ROOT}/controller/task_lora_j.pt"
STRICT_AUDIT="${ROOT}/strict_unseen_18"
STRICT_REPORT="${ROOT}/comparison_strict_unseen_18"
POP_AUDIT="${ROOT}/matched_population_512"
POP_REPORT="${ROOT}/comparison_population_512"
LOG="${LOG_ROOT}/evaluation_recovery.log"
MANIFEST="${ROOT}/run_manifest.json"
mkdir -p "${STRICT_AUDIT}" "${STRICT_REPORT}" "${POP_AUDIT}" "${POP_REPORT}"

"${PYTHON}" - "${MANIFEST}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
d["status"]="evaluating"
d["evaluation_recovery"]={
  "strict_unseen_permutations":18,
  "population_permutations":512,
  "reason":"only 18 permutations remain unseen after long fresh-seed training"
}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${CODE}"
export PYTHONUNBUFFERED=1
cd "${CODE}"
set +e
{
"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_matched_audit \
  --checkpoint "${CHECKPOINT}" \
  --phase-summary "${PHASE}" \
  --before-artifact "${BEFORE}" \
  --after-artifact "${AFTER}" \
  --out-dir "${STRICT_AUDIT}" \
  --device cuda \
  --permutations 18 \
  --batch-size 64 \
  --loops 128 \
  --sample-seed 20260807 \
  --sampling-scope strict_unseen \
  --cuda-memory-fraction 0.16

"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_dense_compare \
  --audit "${STRICT_AUDIT}/summary.json" \
  --out-dir "${STRICT_REPORT}"

"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_matched_audit \
  --checkpoint "${CHECKPOINT}" \
  --phase-summary "${PHASE}" \
  --before-artifact "${BEFORE}" \
  --after-artifact "${AFTER}" \
  --out-dir "${POP_AUDIT}" \
  --device cuda \
  --permutations 512 \
  --batch-size 64 \
  --loops 128 \
  --sample-seed 20260807 \
  --sampling-scope all_permutations \
  --cuda-memory-fraction 0.16

"${PYTHON}" -m reasoning_loop.graph_path_strict_h8_dense_compare \
  --audit "${POP_AUDIT}/summary.json" \
  --out-dir "${POP_REPORT}"
} >"${LOG}" 2>&1
STATUS=$?
set -e
"${PYTHON}" - "${MANIFEST}" "${STATUS}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); code=int(sys.argv[2])
d["evaluation_exit_code"]=code
d["status"]="complete" if code==0 else "failed"
p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY
exit "${STATUS}"
