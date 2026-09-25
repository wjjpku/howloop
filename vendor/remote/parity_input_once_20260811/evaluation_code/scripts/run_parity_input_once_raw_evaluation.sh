#!/usr/bin/env bash
set -euo pipefail

run_root=/data/wujiaju/parity_input_once_20260811
seed="${PARITY_SEED:?set PARITY_SEED}"
code_root="$run_root/evaluation_code"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
out_root="$run_root/evaluation/seed${seed}"
log_file="/data/wujiaju/logs/parity_input_once_20260811/raw_evaluation_seed${seed}.log"
pid_file="$run_root/raw_evaluation_seed${seed}.pid"

mkdir -p "$out_root" "$(dirname "$log_file")"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root"

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CHECKPOINT=$checkpoint"

echo "STAGE far_horizon START $(date --iso-8601=seconds)"
"$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
  --task parity \
  --checkpoint "$checkpoint" \
  --lengths 10 20 24 32 40 48 64 80 100 128 160 200 \
  --examples 512 \
  --max-batch-size 32 \
  --token-budget 8192 \
  --seed "$((2026082101 + seed))" \
  --device cuda \
  --out-dir "$out_root/far_horizon"
echo "STAGE far_horizon COMPLETE $(date --iso-8601=seconds)"

echo "STAGE diagnose START $(date --iso-8601=seconds)"
"$python_bin" -u "$code_root/reasoning_loop/paper_length_telomere.py" diagnose \
  --checkpoint "$checkpoint" \
  --lengths 10 20 24 32 40 48 64 80 100 \
  --batch-size 128 \
  --batches 8 \
  --maximum-step 112 \
  --seed "$((2026082201 + seed))" \
  --device cuda \
  --out-dir "$out_root/diagnose"
echo "STAGE diagnose COMPLETE $(date --iso-8601=seconds)"

echo "STAGE four_phase START $(date --iso-8601=seconds)"
"$python_bin" -u "$code_root/reasoning_loop/analyze_parity_four_phase.py" \
  --checkpoint "$checkpoint" \
  --out-dir "$out_root/four_phase" \
  --device cuda \
  --batch-size 256 \
  --batches 2 \
  --causal-batch-size 256 \
  --discovery-seed "$((2026082301 + seed))" \
  --evaluation-seed "$((2026082401 + seed))" \
  --causal-seed "$((2026082501 + seed))"
echo "STAGE four_phase COMPLETE $(date --iso-8601=seconds)"

echo "STAGE heatmap START $(date --iso-8601=seconds)"
"$python_bin" -u "$code_root/scripts/evaluate_parity_loop_depth_heatmap.py" \
  --checkpoint "$checkpoint" \
  --out-dir "$out_root/loop_depth_heatmap" \
  --min-length 1 \
  --max-length 100 \
  --max-loop 112 \
  --examples 256 \
  --batch-size 64 \
  --seed "$((2026082601 + seed))" \
  --device cuda
echo "STAGE heatmap COMPLETE $(date --iso-8601=seconds)"

echo "COMPLETE $(date --iso-8601=seconds)"
