#!/usr/bin/env bash
# Close an isolated P2 campaign without permitting a partial controller table.
# This is deliberately separate from the asynchronous trainer: a successful
# launch is not evidence, whereas this finalizer leaves audited aggregates and
# figures only after P4 and every registered evaluation have completed.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
namespace="${PAPER2027_PARITY_P2_NAMESPACE:?set the isolated P2 namespace}"
seed="${PAPER2027_PARITY_SEED:-5}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
p4_gpu="${PAPER2027_PARITY_P4_GPU:-3}"
manifest="$run_root/manifests/p2_finalizer_${namespace}_seed${seed}.json"
aggregate_dir="$run_root/aggregate_p2_${namespace}"
figure_dir="$run_root/p2_figures_${namespace}"
audit_attestation="$run_root/audits/p2_p4_${namespace}_seed${seed}.json"

[[ ! -e "$manifest" ]] || { echo "refusing existing finalizer manifest: $manifest" >&2; exit 2; }
[[ ! -e "$audit_attestation" ]] || { echo "refusing existing P2 audit attestation: $audit_attestation" >&2; exit 2; }
for kind in controller evaluation; do
  campaign="$run_root/manifests/p2_${kind}_${namespace}_seed${seed}.json"
  [[ -s "$campaign" ]] || { echo "missing P2 ${kind} campaign manifest" >&2; exit 2; }
  "$python_bin" - "$campaign" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
labels = ("rank48_seed1", "rank48_seed2", "rank128_seed1", "rank128_seed2", "dense_seed1", "dense_seed2")
# The manifest may record the parallel execution order; ordering itself has no
# inferential meaning.  What must be invariant is the exact, duplicate-free
# preregistered label set.
seen = tuple(data.get("labels", ()))
if data.get("status") != "complete" or len(seen) != len(labels) or set(seen) != set(labels):
    raise SystemExit("P2 campaign is incomplete or its registered label set disagrees")
PY
done
for dir in "$aggregate_dir" "$figure_dir"; do
  if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing nonempty finalizer output: $dir" >&2; exit 2
  fi
done
mkdir -p "$(dirname "$manifest")"
"$python_bin" - "$manifest" "$namespace" "$seed" "$aggregate_dir" "$figure_dir" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, namespace, seed, aggregate, figures = sys.argv[1:]
Path(path).write_text(json.dumps({
  "status": "running", "protocol_id": "paper2027.parity.p2.parallel.finalizer.v1",
  "namespace": namespace, "backbone_seed": int(seed), "aggregate_dir": aggregate,
  "figure_dir": figures, "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n")
PY
finish() {
  local exit_status=$?
  "$python_bin" - "$manifest" "$exit_status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
data = json.load(open(path))
data["status"] = "complete" if int(status) == 0 else "failed"
data["updated_at"] = datetime.now(timezone.utc).isoformat()
open(path, "w").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  exit "$exit_status"
}
trap finish EXIT

# P4 is linked to the declared primary rank-48 seed-1 controller; execute it
# only after all P2 data are frozen.  GPU 3 is intentionally outside the
# campaign's 0--2 allocation.
CUDA_VISIBLE_DEVICES="$p4_gpu" PAPER2027_PARITY_ROOT="$run_root" \
PAPER2027_PARITY_P2_NAMESPACE="$namespace" PAPER2027_PARITY_SEED="$seed" \
PAPER2027_PYTHON="$python_bin" bash "$code_root/scripts/run_paper2027_parity_p4_controller_mechanism.sh"

"$python_bin" "$code_root/scripts/audit_paper2027_parity_p2_final.py" \
  --root "$run_root" --namespace "$namespace" --seed "$seed" --out "$audit_attestation"
"$python_bin" "$code_root/scripts/aggregate_paper2027_parity_p2.py" \
  --root "$run_root" --out-dir "$aggregate_dir" --seeds 3 4 "$seed" --namespace "$namespace" \
  --audit-attestation "$audit_attestation"
"$python_bin" "$code_root/scripts/make_paper2027_parity_p2_figures.py" \
  --aggregate-dir "$aggregate_dir" --figure-dir "$figure_dir"
"$python_bin" "$code_root/scripts/make_paper2027_parity_p4_figures.py" \
  --p4-root "$run_root/p4_mechanism_${namespace}/seed${seed}" --figure-dir "$figure_dir"
