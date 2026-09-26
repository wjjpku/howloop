#!/usr/bin/env bash
# Appendix P4: test whether the preregistered rank-48 controller acts through
# the independently fitted phase plane, then localize its matrix components.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
seed="${PAPER2027_PARITY_SEED:?set PAPER2027_PARITY_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/final.pt"
boundary="$run_root/evaluation/seed${seed}/prospective_boundary.json"
namespace="${PAPER2027_PARITY_P2_NAMESPACE:-}"
suffix=""
[[ -z "$namespace" ]] || suffix="_${namespace}"
controller="$run_root/p2_controllers${suffix}/seed${seed}/rank48_seed1/best_controller.pt"
out_dir="$run_root/p4_mechanism${suffix}/seed${seed}"
log="/data/paperexperiment/logs/paper2027_confirmatory/parity_input_once_v2/p4_mechanism_seed${seed}.log"
manifest="$run_root/manifests/p4_mechanism${suffix}_seed${seed}.json"

[[ ! -e "$manifest" ]] || { echo "refusing existing P4 manifest: $manifest" >&2; exit 2; }
if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing nonempty P4 output directory: $out_dir" >&2; exit 2
fi

mkdir -p "$out_dir" "$(dirname "$log")" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$boundary" && -s "$controller" ]] || { echo "missing P4 inputs" >&2; exit 2; }
read -r status minimum maximum < <("$python_bin" - "$boundary" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); print(d["status"], *(d.get("training_range") or (0,0)))
PY
)
[[ "$status" == disease_detected ]] || { echo "NO_DISEASE_SKIP"; exit 0; }

"$python_bin" - "$manifest" "$seed" "$checkpoint" "$controller" "$log" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, seed, checkpoint, controller, log=sys.argv[1:]
Path(path).write_text(json.dumps({"status":"running","updated_at":datetime.now(timezone.utc).isoformat(),"backbone_seed":int(seed),"protocol_id":"paper2027.parity.p4.controller_mechanism.v1","checkpoint":checkpoint,"checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),"controller":controller,"controller_sha256":hashlib.sha256(Path(controller).read_bytes()).hexdigest(),"paper_mode":True,"log":log},indent=2,sort_keys=True)+"\n")
PY
finish() {
  local exit_status=$?
  "$python_bin" - "$manifest" "$exit_status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, exit_status = sys.argv[1:]
payload = json.load(open(path))
payload["status"] = "complete" if int(exit_status) == 0 else "failed"
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
open(path, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  exit "$exit_status"
}
trap finish EXIT
exec >> "$log" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

read -r -a lengths < <("$python_bin" - "$minimum" "$maximum" <<'PY'
import sys
lo, hi = map(int, sys.argv[1:]); print(20, lo, (lo+hi)//2, hi, min(500, 2*hi))
PY
)
"$python_bin" -u "$code_root/reasoning_loop/analyze_parity_four_phase.py" \
  --checkpoint "$checkpoint" --controller "$controller" --paper-mode --out-dir "$out_dir/four_phase" \
  --device cuda --batch-size 256 --batches 2 --causal-batch-size 256 --random-controls 100 \
  --discovery-seed "$((2026095001+seed))" --evaluation-seed "$((2026095101+seed))" --causal-seed "$((2026095201+seed))"
"$python_bin" -u "$code_root/reasoning_loop/evaluate_parity_j_svd_interventions.py" \
  --checkpoint "$checkpoint" --controller "$controller" --paper-mode --out-dir "$out_dir/svd" \
  --device cuda --lengths "${lengths[@]}" --batch-size 64 --batches 4 --post-target-steps 8 --seed "$((2026095301+seed))"
"$python_bin" -u "$code_root/reasoning_loop/analyze_parity_j_hidden_effect.py" \
  --checkpoint "$checkpoint" --controller "$controller" --paper-mode --out-dir "$out_dir/hidden_effect" \
  --device cuda --lengths "${lengths[@]}" --batch-size 32 --batches 2 --seed "$((2026095401+seed))"
