#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ACTION PHYSICAL_GPU" >&2
  exit 2
fi

ACTION="$1"
PHYSICAL_GPU="$2"
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be an explicit physical GPU index" >&2
  exit 2
fi

CODE_ROOT="/data/paperexperiment/LooPlus"
OUTPUT_ROOT="/data/paperexperiment/graph_path_telomere_simple_one_step_D8L8_20260730"
LOG_ROOT="/data/paperexperiment/logs/graph_path_telomere_simple_one_step_D8L8_20260730"
PYTHON_BIN="/data/paperexperiment/.venvs/loopreasoner/bin/python"
CHECKPOINT="/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt"
PHASE_SUMMARY="/data/paperexperiment/graph_path_telomere_overloop_20260729/formal/D8_L8_seed1/summary.json"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TELOMERE_CUDA_MEMORY_FRACTION="${TELOMERE_CUDA_MEMORY_FRACTION:-0.04}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${CODE_ROOT}"
LOG_PATH="${LOG_ROOT}/${ACTION}_gpu${PHYSICAL_GPU}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

for required in "${CHECKPOINT}" "${PHASE_SUMMARY}"; do
  if [[ ! -r "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 3
  fi
done

write_manifest() {
  local status="$1"
  STATUS="${status}" \
  ACTION_NAME="${ACTION}" \
  PHYSICAL_GPU_VALUE="${PHYSICAL_GPU}" \
  LOG_PATH_VALUE="${LOG_PATH}" \
  OUTPUT_ROOT_VALUE="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": os.environ["STATUS"],
    "action": os.environ["ACTION_NAME"],
    "physical_gpu": int(os.environ["PHYSICAL_GPU_VALUE"]),
    "visible_cuda_device": 0,
    "pid": os.getppid(),
    "log_path": os.environ["LOG_PATH_VALUE"],
    "cuda_memory_fraction": float(
        os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
    ),
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = (
    Path(os.environ["OUTPUT_ROOT_VALUE"])
    / f"{os.environ['ACTION_NAME']}_launcher_manifest.json"
)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

run_formal() {
  local label="$1"
  local seed="$2"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_simple_one_step \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches 16 \
    --heldout-batch-size 256 \
    --heldout-batches 8 \
    --evaluation-batch-size 256 \
    --evaluation-batches 8 \
    --extra-loops 16 \
    --ridge 0.01 \
    --seed "${seed}" \
    --device cuda
}

run_circuit() {
  local label="$1"
  local operator="$2"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_simple_one_step_circuit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --operator "${operator}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 4 \
    --cycles 3 \
    --seed 20262730 \
    --device cuda
}

write_manifest "running"
trap 'write_manifest "failed"' ERR

case "${ACTION}" in
  formal)
    run_formal "formal_primary" 20260730
    run_formal "formal_replica" 20261730
    ;;
  circuit)
    run_circuit \
      "circuit_primary" \
      "${OUTPUT_ROOT}/formal_primary/simple_one_step_R.pt"
    run_circuit \
      "circuit_replica" \
      "${OUTPUT_ROOT}/formal_replica/simple_one_step_R.pt"
    ;;
  circuit-fine)
    run_circuit \
      "circuit_fine_primary" \
      "${OUTPUT_ROOT}/formal_primary/simple_one_step_R.pt"
    run_circuit \
      "circuit_fine_replica" \
      "${OUTPUT_ROOT}/formal_replica/simple_one_step_R.pt"
    ;;
  circuit-routing)
    run_circuit \
      "circuit_routing_primary" \
      "${OUTPUT_ROOT}/formal_primary/simple_one_step_R.pt"
    run_circuit \
      "circuit_routing_replica" \
      "${OUTPUT_ROOT}/formal_replica/simple_one_step_R.pt"
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac

trap - ERR
write_manifest "complete"
