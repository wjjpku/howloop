#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t n10d8_seed3_queued 2>/dev/null \
  || tmux has-session -t n10d8_seed4_queued 2>/dev/null \
  || tmux has-session -t n10d8_seed5_queued 2>/dev/null; do
  sleep 20
done

exec /data/wujiaju/LooPlus/scripts/run_graph_n10_d8l8_random_path_readout_20260809.sh \
  0 1 2 3 4 5
