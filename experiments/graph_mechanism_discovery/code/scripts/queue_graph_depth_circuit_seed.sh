#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PREDECESSOR_SESSION GPU MAX_DEPTH LOOPS SEED OUT_DIR LOG_PATH" >&2
  exit 2
fi

predecessor="$1"
shift

while tmux has-session -t "${predecessor}" 2>/dev/null; do
  sleep 20
done

exec /data/wujiaju/LooPlus/scripts/train_graph_depth_circuit_seed.sh "$@"
