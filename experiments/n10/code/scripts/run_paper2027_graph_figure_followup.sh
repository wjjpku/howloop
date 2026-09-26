#!/usr/bin/env bash
# Render Graph figures only after the campaign's fail-closed aggregate exists.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g1_v1}"
code_root="${PAPER2027_GRAPH_CODE_ROOT:-$run_root/analysis_code}"
wait_session="${PAPER2027_GRAPH_WAIT_SESSION:-paper2027_graph_confirmatory_v1}"
sleep_seconds="${PAPER2027_GRAPH_WAIT_SECONDS:-60}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "$sleep_seconds"
done

summary="$run_root/aggregate_confirmatory/summary.json"
[[ -s "$summary" ]] || { echo "missing completed Graph aggregate" >&2; exit 2; }
"$python_bin" - "$summary" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("complete_backbones") != 12:
    raise SystemExit("Graph aggregate does not contain the full preregistered cohort")
PY
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$code_root/scripts/make_paper2027_graph_figures.py" \
  --root "$run_root" --figure-dir "$run_root/paper_figures" \
  --results-tex "$run_root/paper_figures/graph_results.tex" \
  --seeds 100 101 102 103 104 105 106 107 108 109 110 111
