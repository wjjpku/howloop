#!/usr/bin/env bash
set -euo pipefail

repo="${PROJECT_ROOT:-/data/wujiaju/LooPlus}"
python_bin="${PYTHON_BIN:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
checkpoint="${CHECKPOINT:-/data/wujiaju/graph_path_functional_multiseed_20260725/training/D8_L6_seed6/graphpath_N8_D8_d256_B2_L6_seed6/best.pt}"
controllers="${CONTROLLERS:-/data/wujiaju/graph_path_jump_controller_D8L6_20260730/no_prewrite_formal/task_tuned_controllers.pt}"
output_dir="${OUTPUT_DIR:-/data/wujiaju/graph_path_jump_controller_D8L6_20260730/semantic_routing_formal}"

cd "$repo"

exec "$python_bin" -u -m reasoning_loop.graph_path_jump_controller_semantic_routing \
  --checkpoint "$checkpoint" \
  --controller-path "$controllers" \
  --out-dir "$output_dir" \
  --device cuda \
  --controller-seeds 0 1 2 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --eval-batches "${EVAL_BATCHES:-8}" \
  --eval-seed "${EVAL_SEED:-16401}"
