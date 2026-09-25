#!/usr/bin/env bash
# Train capacity-matched P2 controllers only after the raw-only prospective
# boundary rule has fixed the repair band.  A no-disease backbone is recorded
# and skipped; it is never silently removed from the result table.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/wujiaju/paper2027_confirmatory/parity_input_once_v2}"
seed="${PAPER2027_PARITY_SEED:?set PAPER2027_PARITY_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
trainer="$code_root/reasoning_loop/paper_length_telomere.py"
selector="$code_root/reasoning_loop/select_parity_boundary_controller.py"
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/final.pt"
raw_boundary="$run_root/evaluation/seed${seed}/prospective_boundary.json"
namespace="${PAPER2027_PARITY_P2_NAMESPACE:-}"
suffix=""
[[ -z "$namespace" ]] || suffix="_${namespace}"
experiment_root="$run_root/p2_controllers${suffix}/seed${seed}"
log_dir="/data/wujiaju/logs/paper2027_confirmatory/parity_input_once_v2"
labels="${PAPER2027_PARITY_P2_LABELS:-rank48_seed1 rank128_seed1 dense_seed1 rank48_seed2 rank128_seed2 dense_seed2}"
read -r -a label_array <<< "$labels"
[[ ${#label_array[@]} -ge 1 ]] || { echo "P2 needs at least one controller label" >&2; exit 2; }
label_tag="${labels// /_}"
log_file="$log_dir/p2_controller${suffix}_seed${seed}_${label_tag}.log"
manifest="$run_root/manifests/p2_controller${suffix}_seed${seed}_${label_tag}.json"

if [[ -e "$manifest" ]]; then
  echo "refusing existing P2 controller manifest: $manifest" >&2; exit 2
fi
mkdir -p "$experiment_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$raw_boundary" && -s "$trainer" ]] || {
  echo "P2 needs checkpoint, frozen code, and completed raw boundary selection" >&2; exit 2;
}

selection_status="$("$python_bin" - "$raw_boundary" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
range_values="$("$python_bin" - "$raw_boundary" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if data["status"] == "disease_detected": print(*data["training_range"])
PY
)"

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$raw_boundary" "$log_file" "$selection_status" "$range_values" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

path, status, seed, checkpoint, boundary, log, selection_status, *live_range = sys.argv[1:]
payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getpid(),
    "backbone_seed": int(seed),
    "protocol_id": "paper2027.parity.p2.prospective_boundary.v1",
    "checkpoint": checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "raw_boundary_selection": boundary,
    "raw_boundary_sha256": hashlib.sha256(Path(boundary).read_bytes()).hexdigest(),
    "selection_status": selection_status,
    "training_range": [int(v) for v in live_range[0].split()] if live_range else None,
    "paper_mode": True,
    "log": log,
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

if [[ "$selection_status" != "disease_detected" ]]; then
  write_manifest no_disease_skip
  echo "NO_DISEASE_SKIP $(date --iso-8601=seconds)"
  exit 0
fi
read -r minimum_length maximum_length <<< "$range_values"
[[ "$minimum_length" =~ ^[0-9]+$ && "$maximum_length" =~ ^[0-9]+$ ]] || {
  echo "invalid prospective training range: $range_values" >&2; exit 2;
}

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed BOUNDARY=$minimum_length..$maximum_length"

train_and_select() {
  local label="$1"
  local parameterization="$2"
  local rank="$3"
  local controller_seed="$4"
  local out_dir="$experiment_root/$label"
  local parameter_args=(--controller-parameterization "$parameterization")
  if [[ "$parameterization" == "diagonal_low_rank" ]]; then
    parameter_args+=(--rank "$rank")
  fi
  if [[ -s "$out_dir/selection.json" && -s "$out_dir/best_controller.pt" ]]; then
    echo "STAGE $label ALREADY_COMPLETE $(date --iso-8601=seconds)"; return
  fi
  echo "STAGE $label TRAIN_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$trainer" controller \
    --checkpoint "$checkpoint" --paper-mode "${parameter_args[@]}" \
    --seed "$controller_seed" --device cuda --grad-clip 1.0 \
    --learning-rate-multiplier 1.0 --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 --controller-curriculum logical_range \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-logical-min-length "$minimum_length" \
    --controller-logical-max-length "$maximum_length" \
    --controller-anchor-step 1 --controller-ce-temperature 4.0 \
    --stage-round-multiplier 3 --controller-warmup-updates 512 \
    --controller-stable-updates 4096 --controller-lr-schedule wsd \
    --controller-final-lr-ratio 0.1 --controller-checkpoint-every 384 \
    --out-dir "$out_dir"
  echo "STAGE $label SELECT_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$selector" --checkpoint "$checkpoint" --paper-mode \
    --controller-dir "$out_dir" --logical-min-length "$minimum_length" \
    --logical-max-length "$maximum_length" --validation-seed "$((2026090101 + seed))" \
    --examples-per-length 512 --batch-size 64 --retention-tolerance 0.005 \
    --device cuda --out-dir "$out_dir"
  echo "STAGE $label COMPLETE $(date --iso-8601=seconds)"
}

base_controller_seed=$((2026091001 + 100 * seed))
labels="${PAPER2027_PARITY_P2_LABELS:-rank48_seed1 rank128_seed1 dense_seed1 rank48_seed2 rank128_seed2 dense_seed2}"
for label in $labels; do
  target="$experiment_root/$label"
  if [[ -d "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing nonempty P2 controller label output: $target" >&2; exit 2
  fi
  case "$label" in
    rank48_seed1) train_and_select "$label" diagonal_low_rank 48 "$((base_controller_seed + 11))" ;;
    rank128_seed1) train_and_select "$label" diagonal_low_rank 128 "$((base_controller_seed + 12))" ;;
    dense_seed1) train_and_select "$label" dense_affine 0 "$((base_controller_seed + 13))" ;;
    rank48_seed2) train_and_select "$label" diagonal_low_rank 48 "$((base_controller_seed + 21))" ;;
    rank128_seed2) train_and_select "$label" diagonal_low_rank 128 "$((base_controller_seed + 22))" ;;
    dense_seed2) train_and_select "$label" dense_affine 0 "$((base_controller_seed + 23))" ;;
    *) echo "unknown P2 controller label: $label" >&2; exit 2 ;;
  esac
done

echo "COMPLETE $(date --iso-8601=seconds)"
