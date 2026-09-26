#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 PREDECESSOR_SESSION GPU OUT_DIR LOG_PATH NAME=CHECKPOINT [...]" >&2
  exit 2
fi

predecessor="$1"
shift

while tmux has-session -t "${predecessor}" 2>/dev/null; do
  sleep 20
done

exec /data/paperexperiment/LooPlus/scripts/run_graph_depth_circuit_analysis.sh "$@"
