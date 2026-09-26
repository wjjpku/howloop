#!/usr/bin/env bash
set -euo pipefail

run_root=/data/paperexperiment/parity_input_once_20260811
experiment_root="$run_root/boundary_rank_sweep_v1"
seed="${PARITY_SEED:?set PARITY_SEED to 0, 1, or 2}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code_root="$run_root/evaluation_code"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
far_horizon="$code_root/scripts/evaluate_parity_far_horizon.py"
diagonal_band="$code_root/scripts/evaluate_parity_diagonal_band.py"
ridge_slope="$code_root/scripts/analyze_parity_ridge_slope.py"
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
controller="$experiment_root/seed${seed}/boundary_rank48/best_controller.pt"
out_dir="$experiment_root/evaluations/seed${seed}/boundary_rank48"
log_dir=/data/paperexperiment/logs/parity_input_once_20260811/boundary_rank_sweep_v1
log_file="$log_dir/evaluation_rank48_seed${seed}.log"
pid_file="$experiment_root/evaluation_rank48_seed${seed}.pid"
manifest="$experiment_root/evaluation_rank48_seed${seed}_manifest.json"
evaluation_seed=$((2026086001 + seed))

case "$seed" in
  0) minimum_length=80; maximum_length=200 ;;
  1) minimum_length=64; maximum_length=128 ;;
  2) minimum_length=48; maximum_length=100 ;;
  *) echo "PARITY_SEED must be 0, 1, or 2" >&2; exit 2 ;;
esac
ood_minimum=$((maximum_length + 1))

mkdir -p "$out_dir" "$log_dir"
test -s "$controller"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$minimum_length" "$maximum_length" "$evaluation_seed" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, seed, minimum, maximum, evaluation_seed = sys.argv[1:]
payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getppid(),
    "backbone_seed": int(seed),
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "controller": "boundary_rank48",
    "controller_training_range": [int(minimum), int(maximum)],
    "strict_controller_ood_range_for_ridge_fit": [int(maximum) + 1, 490],
    "evaluation_seed": int(evaluation_seed),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_manifest running
echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -s "$out_dir/far_horizon/summary.json" ]]; then
  echo "STAGE FAR_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$far_horizon" \
    --task parity \
    --checkpoint "$checkpoint" \
    --controller "$controller" \
    --lengths 10 20 40 48 64 80 100 128 160 200 250 300 400 500 600 750 1000 \
    --examples 256 \
    --max-batch-size 64 \
    --token-budget 32768 \
    --seed "$evaluation_seed" \
    --device cuda \
    --no-post-final-controller \
    --out-dir "$out_dir/far_horizon"
  echo "STAGE FAR_COMPLETE $(date --iso-8601=seconds)"
fi

if [[ ! -s "$out_dir/diagonal_band/diagonal_band_metrics.csv" ]]; then
  echo "STAGE DIAGONAL_START $(date --iso-8601=seconds)"
  "$python_bin" -u "$diagonal_band" \
    --checkpoint "$checkpoint" \
    --controller "$controller" \
    --out-dir "$out_dir/diagonal_band" \
    --min-length 1 \
    --max-length 500 \
    --length-step 1 \
    --max-loop 510 \
    --half-width 10 \
    --examples 64 \
    --batch-size 32 \
    --seed "$evaluation_seed" \
    --device cuda
  echo "STAGE DIAGONAL_COMPLETE $(date --iso-8601=seconds)"
fi

if [[ ! -s "$out_dir/ridge_supported/parity_ridge_slope_analysis.json" ]]; then
  "$python_bin" -u "$ridge_slope" \
    --metrics "$out_dir/diagonal_band/diagonal_band_metrics.csv" \
    --out-dir "$out_dir/ridge_supported" \
    --minimum-length "$minimum_length" \
    --maximum-length "$maximum_length" \
    --period 4.0 \
    --bootstrap-samples 2000 \
    --bootstrap-block-length 20 \
    --seed "$evaluation_seed"
fi

if [[ ! -s "$out_dir/ridge_ood/parity_ridge_slope_analysis.json" ]]; then
  "$python_bin" -u "$ridge_slope" \
    --metrics "$out_dir/diagonal_band/diagonal_band_metrics.csv" \
    --out-dir "$out_dir/ridge_ood" \
    --minimum-length "$ood_minimum" \
    --maximum-length 490 \
    --period 4.0 \
    --bootstrap-samples 2000 \
    --bootstrap-block-length 20 \
    --seed "$evaluation_seed"
fi

write_manifest complete
echo "COMPLETE $(date --iso-8601=seconds)"
