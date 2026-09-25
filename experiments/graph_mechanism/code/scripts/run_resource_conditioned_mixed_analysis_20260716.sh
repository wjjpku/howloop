#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX nomode|mode SEED [SEED ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
shift 2
case "$arm" in
  nomode|mode) ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
run_root=${RESOURCE_REUSE_RUN_ROOT:-/data/wujiaju/resource_conditioned_reuse_20260716/triadic_mixed/runs}
out_root=${RESOURCE_REUSE_OUT_ROOT:-/data/wujiaju/resource_conditioned_reuse_20260716/triadic_mixed/analysis}
log_root=${RESOURCE_REUSE_LOG_ROOT:-/data/wujiaju/logs/resource_conditioned_reuse_20260716}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
mkdir -p "$out_root" "$log_root"
cd "$repo_root"

for seed in "$@"; do
  run_name="mixed_${arm}_d64_L6_seed${seed}"
  checkpoint="$run_root/$run_name/final.pt"
  output="$out_root/$run_name"
  log_path="$log_root/${run_name}_analysis.log"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "analyzing $run_name on physical GPU $gpu_index"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" - "$checkpoint" "$output" >"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

import torch

from reasoning_loop.triadic_mixed_schedule_diagnostics import analyze_mixed_checkpoint

result = analyze_mixed_checkpoint(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    device=torch.device("cuda"),
    sample_size=1024,
    batch_size=1024,
)
print(json.dumps(result, indent=2))
PY
done
