#!/usr/bin/env bash
# Run the controller-mechanism appendix only after the prespecified P2
# population is complete.  A no-disease result remains an explicit no-op;
# a failed P2 run is an error, never an eligibility filter.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
seed="${PAPER2027_PARITY_P4_SEED:-5}"
wait_session="${PAPER2027_PARITY_WAIT_SESSION:-paper2027_parity_p1_eval_p2_v2}"
sleep_seconds="${PAPER2027_PARITY_WAIT_SECONDS:-60}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
controller_manifest="$run_root/manifests/p2_controller_seed${seed}.json"
evaluation_manifest="$run_root/manifests/p2_evaluation_seed${seed}.json"
runner="$code_root/scripts/run_paper2027_parity_p4_controller_mechanism.sh"

[[ -x "$runner" ]] || { echo "missing P4 runner: $runner" >&2; exit 2; }
while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "$sleep_seconds"
done

[[ -s "$controller_manifest" && -s "$evaluation_manifest" ]] || {
  echo "missing P2 manifests for P4 seed=$seed" >&2; exit 2;
}
controller_status="$("$python_bin" - "$controller_manifest" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
evaluation_status="$("$python_bin" - "$evaluation_manifest" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
if [[ "$controller_status" == "no_disease_skip" && "$evaluation_status" == "no_disease_skip" ]]; then
  echo "P4_NO_DISEASE_SKIP seed=$seed"
  exit 0
fi
[[ "$controller_status" == "complete" && "$evaluation_status" == "complete" ]] || {
  echo "P2 failed or incomplete before P4: controller=$controller_status evaluation=$evaluation_status" >&2
  exit 2
}

PAPER2027_PARITY_ROOT="$run_root" \
PAPER2027_PARITY_SEED="$seed" \
PAPER2027_PYTHON="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}" \
  bash "$runner"
