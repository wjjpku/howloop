#!/usr/bin/env bash
set -euo pipefail

run_root=/data/paperexperiment/parity_input_once_20260811
seed="${PARITY_SEED:?set PARITY_SEED}"
code_root="$run_root/evaluation_code"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
out_dir="$run_root/input_path_ablation/seed${seed}"
log_file="/data/paperexperiment/logs/parity_input_once_20260811/input_path_ablation_seed${seed}.log"
pid_file="$run_root/input_path_ablation_seed${seed}.pid"

mkdir -p "$out_dir" "$(dirname "$log_file")"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root"

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$python_bin" -u "$code_root/scripts/analyze_parity_input_path_ablation.py" \
  --checkpoint "$checkpoint" \
  --out-dir "$out_dir" \
  --lengths 20 24 32 40 64 100 \
  --relative-start -4 \
  --relative-end 8 \
  --examples 512 \
  --batch-size 64 \
  --seed "$((2026082701 + seed))" \
  --device cuda
echo "COMPLETE $(date --iso-8601=seconds)"
