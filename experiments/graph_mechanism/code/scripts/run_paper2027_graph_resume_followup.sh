#!/usr/bin/env bash
# Resume the G3 campaign after externally started, manifest-compatible fits
# have closed.  This only accepts complete manifests, then relies on the
# campaign's idempotent runners to skip existing controller artifacts and
# perform the still-missing locked evaluation and aggregation.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/wujiaju/paper2027_confirmatory/graph_g1_v1}"
code_root="${PAPER2027_GRAPH_CODE_ROOT:-$run_root/analysis_code}"
sleep_seconds="${PAPER2027_GRAPH_WAIT_SECONDS:-60}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"

while tmux has-session -t paper2027_g3_seed109 2>/dev/null || tmux has-session -t paper2027_g3_seed110 2>/dev/null; do
  sleep "$sleep_seconds"
done
for seed in 100 101 102 103 104 105 106 107 108 109 110 111; do
  manifest="$run_root/manifests/g3_controller_seed${seed}.json"
  [[ -s "$manifest" ]] || { echo "missing G3 manifest for seed=$seed" >&2; exit 2; }
  status="$($python_bin - "$manifest" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
  [[ "$status" == complete || "$status" == endpoint_failed ]] || {
    echo "G3 seed=$seed did not close cleanly: $status" >&2; exit 2;
  }
done
cd "$code_root"
PAPER2027_GRAPH_ROOT="$run_root" \
PAPER2027_GRAPH_GPUS="3 4 5" \
PAPER2027_PYTHON="$python_bin" \
  bash ./scripts/run_paper2027_graph_g3_campaign.sh
