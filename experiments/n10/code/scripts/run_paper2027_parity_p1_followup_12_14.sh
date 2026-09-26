#!/usr/bin/env bash
# Queue the final independent P1 backbone wave only after the initial campaign
# has exited successfully.  This avoids mixing a partial first wave into the
# preregistered population while keeping all physical GPUs isolated.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/code}"
wait_session="${PAPER2027_PARITY_WAIT_SESSION:-paper2027_parity_p1_v2}"
sleep_seconds="${PAPER2027_PARITY_WAIT_SECONDS:-60}"

[[ -x "$code_root/run_paper2027_parity_p1_campaign.sh" ]] || {
  echo "missing P1 campaign runner: $code_root" >&2
  exit 2
}

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "$sleep_seconds"
done

for seed in 3 4 5 6 7 8 9 10 11; do
  manifest="$run_root/manifests/backbone_seed${seed}.json"
  [[ -s "$manifest" ]] && grep -q '"status": "complete"' "$manifest" || {
    echo "P1 prerequisite is incomplete or failed: seed=$seed" >&2
    exit 2
  }
done

cd "$code_root"
PAPER2027_PARITY_SEEDS="12 13 14" \
PAPER2027_PARITY_ROOT="$run_root" \
PAPER2027_PYTHON="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}" \
  bash ./run_paper2027_parity_p1_campaign.sh
