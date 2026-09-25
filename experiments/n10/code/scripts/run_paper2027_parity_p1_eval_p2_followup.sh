#!/usr/bin/env bash
# Finish the preregistered P1 population before evaluating it, then run P2
# only on the three prespecified deep seeds.  Every prerequisite is manifest
# gated so a failed backbone cannot disappear from the population silently.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/wujiaju/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
scripts_dir="$code_root/scripts"
wait_session="${PAPER2027_PARITY_WAIT_SESSION:-paper2027_parity_p1_v2b}"
sleep_seconds="${PAPER2027_PARITY_WAIT_SECONDS:-60}"

for runner in \
  run_paper2027_parity_p1_evaluation_campaign.sh \
  run_paper2027_parity_p2_campaign.sh; do
  [[ -x "$scripts_dir/$runner" ]] || {
    echo "missing Parity follow-up runner: $scripts_dir/$runner" >&2
    exit 2
  }
done

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "$sleep_seconds"
done

for seed in 3 4 5 6 7 8 9 10 11 12 13 14; do
  manifest="$run_root/manifests/backbone_seed${seed}.json"
  [[ -s "$manifest" ]] && grep -q '"status": "complete"' "$manifest" || {
    echo "P1 population is incomplete or failed: seed=$seed" >&2
    exit 2
  }
done

cd "$code_root"
PAPER2027_PARITY_ROOT="$run_root" \
PAPER2027_PYTHON="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}" \
  bash ./scripts/run_paper2027_parity_p1_evaluation_campaign.sh

PAPER2027_PARITY_ROOT="$run_root" \
PAPER2027_PYTHON="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}" \
  bash ./scripts/run_paper2027_parity_p2_campaign.sh
