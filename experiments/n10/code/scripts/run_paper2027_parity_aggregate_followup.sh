#!/usr/bin/env bash
# Produce immutable P1/P2 summaries and figures only after their manifest
# gates have closed.  P4 is required only for a disease-positive backbone.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
wait_session="${PAPER2027_PARITY_WAIT_SESSION:-paper2027_parity_p4_v2}"
sleep_seconds="${PAPER2027_PARITY_WAIT_SECONDS:-60}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "$sleep_seconds"
done

for seed in 3 4 5 6 7 8 9 10 11 12 13 14; do
  [[ -s "$run_root/manifests/backbone_seed${seed}.json" ]] || exit 2
  grep -q '"status": "complete"' "$run_root/manifests/backbone_seed${seed}.json" || exit 2
  mode=endpoint
  [[ "$seed" == 3 || "$seed" == 4 || "$seed" == 5 ]] && mode=deep
  [[ -s "$run_root/manifests/evaluation_seed${seed}_${mode}.json" ]] || exit 2
  grep -q '"status": "complete"' "$run_root/manifests/evaluation_seed${seed}_${mode}.json" || exit 2
done

for seed in 3 4 5; do
  [[ -s "$run_root/manifests/p2_controller_seed${seed}.json" ]] || exit 2
  [[ -s "$run_root/manifests/p2_evaluation_seed${seed}.json" ]] || exit 2
done

# A disease-positive P2 seed must have completed the corresponding P4
# mechanism run.  This is an audit check, not eligibility-based selection.
for seed in 3 4 5; do
  p2_status="$($python_bin - "$run_root/manifests/p2_controller_seed${seed}.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
  if [[ "$p2_status" == complete ]]; then
    [[ -s "$run_root/manifests/p4_mechanism_seed${seed}.json" ]] || exit 2
    grep -q '"status": "complete"' "$run_root/manifests/p4_mechanism_seed${seed}.json" || exit 2
  elif [[ "$p2_status" != no_disease_skip ]]; then
    exit 2
  fi
done

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$code_root/scripts/aggregate_paper2027_parity_p1.py" \
  --root "$run_root" --out-dir "$run_root/aggregate_p1" \
  --seeds 3 4 5 6 7 8 9 10 11 12 13 14 --deep-seeds 3 4 5 \
  --bootstrap-draws 10000 --seed 2026099001
"$python_bin" "$code_root/scripts/make_paper2027_parity_p1_figures.py" \
  --aggregate-dir "$run_root/aggregate_p1" --figure-dir "$run_root/paper_figures" \
  --results-tex "$run_root/paper_figures/parity_p1_results.tex"
"$python_bin" "$code_root/scripts/aggregate_paper2027_parity_p2.py" \
  --root "$run_root" --out-dir "$run_root/aggregate_p2" --seeds 3 4 5
"$python_bin" "$code_root/scripts/make_paper2027_parity_p2_figures.py" \
  --aggregate-dir "$run_root/aggregate_p2" --figure-dir "$run_root/paper_figures"
