#!/usr/bin/env bash
set -euo pipefail

repo="${PROJECT_ROOT:-/data/paperexperiment/LooPlus}"
python_bin="${PYTHON_BIN:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
checkpoint="${CHECKPOINT:-/data/paperexperiment/graph_path_functional_multiseed_20260725/training/D8_L6_seed6/graphpath_N8_D8_d256_B2_L6_seed6/best.pt}"
output_dir="${OUTPUT_DIR:-/data/paperexperiment/graph_path_jump_controller_D8L6_20260730/formal}"

cd "$repo"

"$python_bin" -u -m reasoning_loop.graph_path_jump_controller \
  --checkpoint "$checkpoint" \
  --out-dir "$output_dir" \
  --device cuda \
  --calibration-batch-size "${CALIBRATION_BATCH_SIZE:-128}" \
  --calibration-batches "${CALIBRATION_BATCHES:-8}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --eval-batches "${EVAL_BATCHES:-8}" \
  --ranks 0 8 16 32 64 128 256 \
  --ridge 1e-3 \
  --seed 6301 \
  --eval-seed 6401
