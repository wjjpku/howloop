#!/usr/bin/env bash
set -euo pipefail
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_graph_g1_evaluation.sh"
run_root="${PAPER2027_GRAPH_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g1_v1}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
seeds="${PAPER2027_GRAPH_SEEDS:-100 101 102 103 104 105 106 107 108 109 110 111}"
gpus="${PAPER2027_GRAPH_GPUS:-0 1 2}"
read -r -a seed_array <<< "$seeds"
read -r -a gpu_array <<< "$gpus"
if (( ${#gpu_array[@]} != 3 )); then
  echo "PAPER2027_GRAPH_GPUS must name exactly three physical GPUs" >&2; exit 2
fi
# Create and hash the common test set once before parallel checkpoint reads.
# This avoids a concurrent first-write race and preserves its locked status.
PAPER2027_GRAPH_ROOT="$run_root" PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - <<'PY'
import os
from pathlib import Path
from reasoning_loop.graph_path_loop import GraphPathConfig
from scripts.evaluate_paper2027_graph_g1 import make_or_load_locked_test

root = Path(os.environ["PAPER2027_GRAPH_ROOT"])
cfg = GraphPathConfig(
    node_count=8, max_depth=8, d_model=256, n_heads=4, d_mlp=1024,
    n_layers=2, max_loops=8, inner_norm_style="pre_layernorm",
    block_schedule="all_blocks",
)
make_or_load_locked_test(
    cfg=cfg,
    path=root / "locked/graph_permutations_512_all_starts.pt",
    permutations=512,
    seed=2026093001,
)
PY
for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for slot in 0 1 2; do
    index=$((start + slot)); (( index < ${#seed_array[@]} )) || break
    CUDA_VISIBLE_DEVICES="${gpu_array[slot]}" PAPER2027_GRAPH_SEED="${seed_array[index]}" bash "$runner" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
