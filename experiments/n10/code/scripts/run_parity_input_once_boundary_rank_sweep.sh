#!/usr/bin/env bash
set -euo pipefail

run_root=/data/paperexperiment/parity_input_once_20260811
experiment_root="$run_root/boundary_rank_sweep_v1"
seed="${PARITY_SEED:?set PARITY_SEED to 0, 1, or 2}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code_root="$run_root/evaluation_code"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
trainer="$code_root/reasoning_loop/paper_length_telomere.py"
selector="$code_root/reasoning_loop/select_parity_boundary_controller.py"
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
log_dir=/data/paperexperiment/logs/parity_input_once_20260811/boundary_rank_sweep_v1
log_file="$log_dir/seed${seed}.log"
pid_file="$experiment_root/seed${seed}.pid"
manifest="$experiment_root/seed${seed}_manifest.json"
controller_seed=$((411001 + seed))
validation_seed=$((2026085001 + seed))

case "$seed" in
  0) minimum_length=80; maximum_length=200 ;;
  1) minimum_length=64; maximum_length=128 ;;
  2) minimum_length=48; maximum_length=100 ;;
  *) echo "PARITY_SEED must be 0, 1, or 2" >&2; exit 2 ;;
esac

mkdir -p "$experiment_root" "$log_dir"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$minimum_length" "$maximum_length" "$controller_seed" "$validation_seed" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, seed, minimum, maximum, controller_seed, validation_seed = sys.argv[1:]
payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getppid(),
    "backbone_seed": int(seed),
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "boundary_training_range": [int(minimum), int(maximum)],
    "controller_ood_definition": f"n>{maximum}",
    "controller_seed": int(controller_seed),
    "validation_seed": int(validation_seed),
    "parameterizations": ["rank48", "rank128", "dense"],
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

train_and_select() {
  local label="$1"
  local parameterization="$2"
  local rank="$3"
  local live_minimum="${4:-$minimum_length}"
  local live_maximum="${5:-$maximum_length}"
  local out_dir="$experiment_root/seed${seed}/${label}"
  local parameter_args=(--controller-parameterization "$parameterization")
  if [[ "$parameterization" == "diagonal_low_rank" ]]; then
    parameter_args+=(--rank "$rank")
  fi

  if [[ -s "$out_dir/selection.json" && -s "$out_dir/best_controller.pt" ]]; then
    echo "STAGE $label ALREADY_COMPLETE $(date --iso-8601=seconds)"
    return
  fi
  echo "STAGE $label TRAIN_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$trainer" controller \
    --checkpoint "$checkpoint" \
    --paper-mode \
    "${parameter_args[@]}" \
    --seed "$controller_seed" \
    --device cuda \
    --grad-clip 1.0 \
    --learning-rate-multiplier 1.0 \
    --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 \
    --controller-curriculum logical_range \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-logical-min-length "$live_minimum" \
    --controller-logical-max-length "$live_maximum" \
    --controller-anchor-step 1 \
    --controller-ce-temperature 4.0 \
    --stage-round-multiplier 3 \
    --controller-warmup-updates 512 \
    --controller-stable-updates 4096 \
    --controller-lr-schedule wsd \
    --controller-final-lr-ratio 0.1 \
    --controller-checkpoint-every 384 \
    --out-dir "$out_dir"
  echo "STAGE $label TRAIN_COMPLETE $(date --iso-8601=seconds)"

  echo "STAGE $label SELECT_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$selector" \
    --checkpoint "$checkpoint" \
    --controller-dir "$out_dir" \
    --logical-min-length "$live_minimum" \
    --logical-max-length "$live_maximum" \
    --validation-seed "$validation_seed" \
    --examples-per-length 512 \
    --batch-size 64 \
    --retention-tolerance 0.005 \
    --device cuda \
    --out-dir "$out_dir"
  echo "STAGE $label COMPLETE $(date --iso-8601=seconds)"
}

write_manifest running
echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CHECKPOINT=$checkpoint"
echo "BOUNDARY_RANGE=$minimum_length..$maximum_length"

train_and_select "boundary_rank48" "diagonal_low_rank" 48
train_and_select "boundary_rank128" "diagonal_low_rank" 128
train_and_select "boundary_dense" "dense_affine" 0
if [[ "$seed" == "1" ]]; then
  train_and_select "easy20to40_rank128" "diagonal_low_rank" 128 20 40
fi

write_manifest complete
echo "COMPLETE $(date --iso-8601=seconds)"
