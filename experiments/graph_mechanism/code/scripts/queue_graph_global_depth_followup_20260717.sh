#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 6 ]]; then
  echo "usage: $0 WAIT_SESSION GPU ANALYSIS_ARM ANALYSIS_SEED NEXT_ARM NEXT_SEED" >&2
  exit 2
fi

wait_session=$1
gpu_index=$2
analysis_arm=$3
analysis_seed=$4
next_arm=$5
next_seed=$6
repo_root=${GLOBAL_DEPTH_REPO_ROOT:-/data/wujiaju/LooPlus}

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep 15
done

cd "$repo_root"
./scripts/run_graph_global_depth_analysis_20260717.sh \
  "$gpu_index" "$analysis_arm" "$analysis_seed"

idle_samples=0
while (( idle_samples < 3 )); do
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

exec ./scripts/run_graph_global_depth_20260717.sh \
  "$gpu_index" "$next_arm" "$next_seed"
