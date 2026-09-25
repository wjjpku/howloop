#!/usr/bin/env bash
# Repair only the missing no-disease *ledger* entries from a failed P2 final
# audit, then re-run the immutable audit/aggregate/figure closure.  This does
# not train, evaluate, select, or alter the disease-positive seed; it invokes
# the existing evaluator only for the two already-recorded no-disease gates,
# whose documented behavior is to write a skip manifest and exit before CUDA.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/wujiaju/paper2027_confirmatory/parity_input_once_v2}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
namespace="${PAPER2027_PARITY_P2_NAMESPACE:?set the isolated P2 namespace}"
seed="${PAPER2027_PARITY_SEED:-5}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
recovery="${PAPER2027_PARITY_P2_RECOVERY_ID:-ledger_retry1}"
previous="$run_root/manifests/p2_finalizer_${namespace}_seed${seed}.json"
manifest="$run_root/manifests/p2_finalizer_${namespace}_seed${seed}_${recovery}.json"
attestation="$run_root/audits/p2_p4_${namespace}_seed${seed}_${recovery}.json"
aggregate_dir="$run_root/aggregate_p2_${namespace}_${recovery}"
figure_dir="$run_root/p2_figures_${namespace}_${recovery}"
evaluator="$code_root/scripts/run_paper2027_parity_p2_evaluation.sh"

[[ -s "$previous" && ! -e "$manifest" && ! -e "$attestation" ]] || {
  echo "recovery requires one historical failed finalizer and fresh recovery outputs" >&2; exit 2;
}
"$python_bin" - "$previous" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("status") != "failed":
    raise SystemExit("recovery is permitted only after a failed finalizer")
PY
for directory in "$aggregate_dir" "$figure_dir"; do
  if [[ -d "$directory" ]] && [[ -n "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing nonempty recovery output: $directory" >&2; exit 2
  fi
done
mkdir -p "$(dirname "$manifest")"
"$python_bin" - "$manifest" "$namespace" "$seed" "$recovery" "$previous" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, namespace, seed, recovery, previous = sys.argv[1:]
Path(path).write_text(json.dumps({
  "status":"running", "protocol_id":"paper2027.parity.p2.audit_ledger_recovery.v1",
  "namespace":namespace, "backbone_seed":int(seed), "recovery_id":recovery,
  "previous_finalizer":previous,
  "scope":"write missing no-disease evaluator manifests only; then re-audit immutable P2/P4 outputs",
  "updated_at":datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True)+"\n")
PY
finish() {
  local status=$?
  "$python_bin" - "$manifest" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
d=json.load(open(path)); d["status"]="complete" if int(status)==0 else "failed"
d["updated_at"]=datetime.now(timezone.utc).isoformat()
open(path,"w").write(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY
  exit "$status"
}
trap finish EXIT

for no_disease_seed in 3 4; do
  boundary="$run_root/evaluation/seed${no_disease_seed}/prospective_boundary.json"
  expected="$run_root/manifests/p2_evaluation_seed${no_disease_seed}.json"
  [[ ! -e "$expected" ]] || { echo "skip manifest unexpectedly already exists: $expected" >&2; exit 2; }
  "$python_bin" - "$boundary" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("status") != "no_disease_detected":
    raise SystemExit("recovery may only write an already-recorded no-disease skip")
PY
  # These historical skip manifests intentionally have no P2 capacity
  # namespace: they describe the original prospective gate, not a controller
  # worker in the isolated positive-seed campaign.
  env -u PAPER2027_PARITY_P2_NAMESPACE CUDA_VISIBLE_DEVICES="" \
    PAPER2027_PARITY_ROOT="$run_root" PAPER2027_PARITY_SEED="$no_disease_seed" \
    PAPER2027_PYTHON="$python_bin" bash "$evaluator"
  "$python_bin" - "$expected" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
if d.get("status") != "no_disease_skip":
    raise SystemExit("evaluator did not write an explicit no-disease skip")
PY
done

"$python_bin" "$code_root/scripts/audit_paper2027_parity_p2_final.py" \
  --root "$run_root" --namespace "$namespace" --seed "$seed" --out "$attestation"
"$python_bin" "$code_root/scripts/aggregate_paper2027_parity_p2.py" \
  --root "$run_root" --out-dir "$aggregate_dir" --seeds 3 4 "$seed" --namespace "$namespace" \
  --audit-attestation "$attestation"
"$python_bin" "$code_root/scripts/make_paper2027_parity_p2_figures.py" \
  --aggregate-dir "$aggregate_dir" --figure-dir "$figure_dir"
"$python_bin" "$code_root/scripts/make_paper2027_parity_p4_figures.py" \
  --p4-root "$run_root/p4_mechanism_${namespace}/seed${seed}" --figure-dir "$figure_dir"
