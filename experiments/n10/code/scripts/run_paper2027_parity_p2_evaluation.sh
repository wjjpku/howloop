#!/usr/bin/env bash
# Evaluate each selected P2 controller on raw-matched samples.  Rank-48 is
# the preregistered primary factorized controller; rank-128 and dense are
# capacity controls.  Component views are retained as appendix-only evidence.
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
controller_root="$run_root/p2_controllers${suffix}/seed${seed}"
out_root="$run_root/p2_evaluation${suffix}/seed${seed}"
log_dir="/data/paperexperiment/logs/paper2027_confirmatory/parity_input_once_v2"
labels="${PAPER2027_PARITY_P2_LABELS:-rank48_seed1 rank48_seed2 rank128_seed1 rank128_seed2 dense_seed1 dense_seed2}"
read -r -a label_array <<< "$labels"
[[ ${#label_array[@]} -ge 1 ]] || { echo "P2 needs at least one evaluation label" >&2; exit 2; }
label_tag="${labels// /_}"
log_file="$log_dir/p2_evaluation${suffix}_seed${seed}_${label_tag}.log"
manifest="$run_root/manifests/p2_evaluation${suffix}_seed${seed}_${label_tag}.json"

if [[ -e "$manifest" ]]; then
  echo "refusing existing P2 evaluation manifest: $manifest" >&2; exit 2
fi
mkdir -p "$out_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$boundary" ]] || { echo "missing P2 inputs" >&2; exit 2; }
read -r selection_status minimum_length maximum_length < <("$python_bin" - "$boundary" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data["status"], *(data.get("training_range") or (0, 0)))
PY
)

# A non-eligible backbone has no controller-label dimension.  Its skip is a
# gate-level result and must use one canonical manifest even when a caller
# inherited a parallel-worker label list.
if [[ "$selection_status" != disease_detected ]]; then
  manifest="$run_root/manifests/p2_evaluation_seed${seed}.json"
  if [[ -e "$manifest" ]]; then
    echo "refusing existing no-disease P2 evaluation manifest: $manifest" >&2; exit 2
  fi
fi

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$boundary" "$log_file" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, seed, checkpoint, boundary, log = sys.argv[1:]
payload = {
    "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
    "backbone_seed": int(seed), "protocol_id": "paper2027.parity.p2.evaluation.v1",
    "checkpoint": checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "raw_boundary_selection": boundary,
    "raw_boundary_sha256": hashlib.sha256(Path(boundary).read_bytes()).hexdigest(),
    "paper_mode": True, "log": log,
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

if [[ "$selection_status" != disease_detected ]]; then
  write_manifest no_disease_skip
  echo "NO_DISEASE_SKIP $(date --iso-8601=seconds)"
  exit 0
fi

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t lengths < <("$python_bin" - "$minimum_length" "$maximum_length" <<'PY'
import sys
lo, hi = map(int, sys.argv[1:])
points = [10, 20, lo, (lo + hi) // 2, hi, hi + 1,
          round(1.25 * hi), round(1.5 * hi), 2 * hi, 3 * hi]
print(*sorted({max(1, min(500, int(x))) for x in points}), sep="\n")
PY
)

echo "START $(date --iso-8601=seconds) SEED=$seed BOUNDARY=$minimum_length..$maximum_length"
echo "LENGTHS=${lengths[*]}"

evaluate_full() {
  # Keep assignments separate: with ``set -u`` Bash expands a multi-variable
  # local declaration before binding its first name.
  local label="$1"
  local controller="$controller_root/$label/best_controller.pt"
  [[ -s "$controller" ]] || { echo "missing selected controller: $controller" >&2; exit 2; }
  "$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
    --task parity --checkpoint "$checkpoint" --controller "$controller" --paper-mode \
    --lengths "${lengths[@]}" --examples 512 --max-batch-size 32 --token-budget 8192 \
    --seed "$((2026092001 + seed))" --device cuda --out-dir "$out_root/$label"
}
for label in $labels; do
  target="$out_root/$label"
  if [[ -d "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing nonempty P2 evaluation label output: $target" >&2; exit 2
  fi
  evaluate_full "$label"
done

# In a parallel worker, only full-controller horizons are evaluated.  The
# primary rank-48 worker below additionally owns all component and executor
# controls, so each control has exactly one registered evaluation.
if [[ "${PAPER2027_PARITY_P2_COMPONENTS:-1}" != 1 ]]; then
  echo "COMPLETE $(date --iso-8601=seconds)"; exit 0
fi

primary="$controller_root/rank48_seed1/best_controller.pt"
for mode in no_AB D_only identity_D AB_only mean_D no_bias shuffle_D spectrum_matched_random_delta; do
  "$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
    --task parity --checkpoint "$checkpoint" --controller "$primary" --paper-mode \
    --controller-mode "$mode" --lengths "${lengths[@]}" \
    --examples 512 --max-batch-size 32 --token-budget 8192 \
    --seed "$((2026092101 + seed))" --device cuda \
    --out-dir "$out_root/rank48_seed1_components/$mode"
done

# A controller without the frozen executor cannot be interpreted as recurrent
# repair; this is a required negative control, evaluated on the same samples.
"$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
  --task parity --checkpoint "$checkpoint" --controller "$primary" --paper-mode \
  --executor-off-after-anchor --lengths "${lengths[@]}" \
  --examples 512 --max-batch-size 32 --token-budget 8192 \
  --seed "$((2026092101 + seed))" --device cuda \
  --out-dir "$out_root/rank48_seed1_executor_off"

"$python_bin" -u "$code_root/scripts/evaluate_parity_loop_depth_heatmap.py" \
  --checkpoint "$checkpoint" --controller "$primary" --paper-mode \
  --out-dir "$out_root/rank48_seed1_loop_depth_heatmap" \
  --min-length 1 --max-length 100 --max-loop 112 --examples 256 --batch-size 64 \
  --seed "$((2026092201 + seed))" --device cuda

echo "COMPLETE $(date --iso-8601=seconds)"
