#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ACTION PHYSICAL_GPU" >&2
  exit 2
fi

ACTION="$1"
PHYSICAL_GPU="$2"
if [[ "${ACTION}" != "formal_screen" ]]; then
  echo "ACTION must be formal_screen" >&2
  exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be an explicit physical GPU index" >&2
  exit 2
fi

CODE_ROOT="/data/paperexperiment/LooPlus"
OUTPUT_ROOT="/data/paperexperiment/graph_path_telomere_graph_blind_mlp_j_20260731"
OUT_DIR="${OUTPUT_ROOT}/${ACTION}"
LOG_ROOT="/data/paperexperiment/logs/graph_path_telomere_graph_blind_mlp_j_20260731"
PYTHON_BIN="/data/paperexperiment/.venvs/loopreasoner/bin/python"
CHECKPOINT="/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt"
PHASE_SUMMARY="/data/paperexperiment/graph_path_telomere_overloop_20260729/formal/D8_L8_seed0/summary.json"
LOG_PATH="${LOG_ROOT}/${ACTION}_gpu${PHYSICAL_GPU}_$(date -u +%Y%m%dT%H%M%SZ).log"
MANIFEST_PATH="${OUTPUT_ROOT}/${ACTION}_launcher_manifest.json"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
cd "${CODE_ROOT}"

for required in "${CHECKPOINT}" "${PHASE_SUMMARY}"; do
  if [[ ! -r "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 3
  fi
done

write_manifest() {
  local status="$1"
  STATUS_VALUE="${status}" \
  ACTION_VALUE="${ACTION}" \
  PHYSICAL_GPU_VALUE="${PHYSICAL_GPU}" \
  LOG_PATH_VALUE="${LOG_PATH}" \
  OUTPUT_PATH_VALUE="${OUT_DIR}" \
  MANIFEST_PATH_VALUE="${MANIFEST_PATH}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": os.environ["STATUS_VALUE"],
    "action": os.environ["ACTION_VALUE"],
    "physical_gpu": int(os.environ["PHYSICAL_GPU_VALUE"]),
    "visible_cuda_device": 0,
    "pid": os.getppid(),
    "log_path": os.environ["LOG_PATH_VALUE"],
    "output_path": os.environ["OUTPUT_PATH_VALUE"],
    "declared_peak_gib": 4.0,
    "smoke_observed_peak_gib": 0.061,
    "reserve_gib": 16.0,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(os.environ["MANIFEST_PATH_VALUE"])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

on_exit() {
  local code="$?"
  if [[ "${code}" -eq 0 ]]; then
    write_manifest "complete"
  else
    write_manifest "failed"
  fi
  exit "${code}"
}
trap on_exit EXIT

write_manifest "running"
exec > >(tee -a "${LOG_PATH}") 2>&1

"${PYTHON_BIN}" -u -m \
  reasoning_loop.graph_path_telomere_graph_blind_mlp_j \
  --checkpoint "${CHECKPOINT}" \
  --phase-summary "${PHASE_SUMMARY}" \
  --out-dir "${OUT_DIR}" \
  --hidden-widths 64 256 512 \
  --training-seeds 190003 290003 390003 \
  --calibration-batch-size 128 \
  --calibration-batches 16 \
  --rounds 24 \
  --training-batch-size 64 \
  --horizons 8 16 24 32 \
  --epochs-per-round 2 \
  --mini-batch-size 256 \
  --learning-rate 5e-5 \
  --evaluation-train-graphs 64 \
  --evaluation-heldout-graphs 64 \
  --evaluation-batch-size 128 \
  --continuation-loops 24 \
  --physical-gpu "${PHYSICAL_GPU}" \
  --prelaunch-used-mib "${PRELAUNCH_USED_MIB:-4}" \
  --prelaunch-free-mib "${PRELAUNCH_FREE_MIB:-81150}" \
  --declared-peak-gib 4 \
  --reserve-gib 16
