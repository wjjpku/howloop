#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_INDEX CONDITION [CONDITION ...]" >&2
  exit 2
fi

gpu_index=$1
shift
repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/paperexperiment/LooPlus}
source_root=${TRIADIC_SOURCE_ROOT:-/data/paperexperiment/triadic_shortage_reuse_20260716/runs}
out_root=${RESOURCE_REUSE_OUT_ROOT:-/data/paperexperiment/resource_conditioned_reuse_20260716/triadic_access}
log_root=${RESOURCE_REUSE_LOG_ROOT:-/data/paperexperiment/logs/resource_conditioned_reuse_20260716}
python_bin=${RESOURCE_REUSE_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}

mkdir -p "$out_root" "$log_root"
cd "$repo_root"

for condition in "$@"; do
  case "$condition" in
    full|full_once|sequential|sequential_shuffled) ;;
    *)
      echo "unknown condition: $condition" >&2
      exit 2
      ;;
  esac
  for seed in 0 1 2; do
    run_name="info_${condition}_d64_L6_seed${seed}"
    checkpoint="$source_root/$run_name/final.pt"
    output="$out_root/$run_name"
    log_path="$log_root/access_${run_name}.log"
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

from reasoning_loop.triadic_access_diagnostics import analyze_checkpoint

checkpoint = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
result = analyze_checkpoint(
    checkpoint,
    out_dir,
    device=torch.device("cuda"),
    sample_size=1024,
    batch_size=1024,
    target_conditions=("full", "full_once", "sequential"),
)
print(json.dumps(result, indent=2))
PY
  done
done
