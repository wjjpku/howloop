#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 WAIT_SESSION GPU COMPLETED_ARM COMPLETED_SEED NEXT_ARM:NEXT_SEED [NEXT_ARM:NEXT_SEED ...]" >&2
  exit 2
fi

wait_session=$1
gpu_index=$2
completed_arm=$3
completed_seed=$4
shift 4
repo_root=${GLOBAL_DEPTH_REPO_ROOT:-/data/paperexperiment/LooPlus}

wait_for_idle_gpu() {
  local idle_samples=0
  while (( idle_samples < 3 )); do
    local utilization process_rows
    utilization=$(
      nvidia-smi -i "$gpu_index" \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits |
        tr -d ' '
    )
    process_rows=$(
      nvidia-smi -i "$gpu_index" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true
    )
    if [[ -z "$process_rows" ]] && (( utilization <= 10 )); then
      idle_samples=$((idle_samples + 1))
    else
      idle_samples=0
    fi
    if (( idle_samples < 3 )); then
      sleep 5
    fi
  done
}

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep 15
done

cd "$repo_root"
./scripts/run_graph_global_depth_analysis_20260717.sh \
  "$gpu_index" "$completed_arm" "$completed_seed"

for spec in "$@"; do
  if [[ "$spec" != *:* ]]; then
    echo "invalid run spec: $spec" >&2
    exit 2
  fi
  arm=${spec%%:*}
  seed=${spec#*:}
  wait_for_idle_gpu
  ./scripts/run_graph_global_depth_20260717.sh "$gpu_index" "$arm" "$seed"
  ./scripts/run_graph_global_depth_analysis_20260717.sh \
    "$gpu_index" "$arm" "$seed"
done
