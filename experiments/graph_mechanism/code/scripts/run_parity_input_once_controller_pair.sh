#!/usr/bin/env bash
set -euo pipefail

run_root=/data/wujiaju/parity_input_once_20260811
seed="${PARITY_SEED:?set PARITY_SEED}"
code_file="$run_root/evaluation_code/reasoning_loop/paper_length_telomere.py"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
log_file="/data/wujiaju/logs/parity_input_once_20260811/controller_pair_seed${seed}.log"
pid_file="$run_root/controller_pair_seed${seed}.pid"

mkdir -p "$(dirname "$log_file")"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1

train_controller() {
  local label="$1"
  local maximum_length="$2"
  local out_dir="$run_root/controllers/seed${seed}/${label}"
  echo "STAGE $label START $(date --iso-8601=seconds)"
  "$python_bin" -u "$code_file" controller \
    --checkpoint "$checkpoint" \
    --controller-parameterization diagonal_low_rank \
    --rank 48 \
    --seed 211001 \
    --device cuda \
    --grad-clip 1.0 \
    --learning-rate-multiplier 1.0 \
    --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-curriculum logical_range \
    --controller-logical-max-length "$maximum_length" \
    --controller-anchor-step 1 \
    --controller-warmup-updates 1024 \
    --controller-final-lr-ratio 0.1 \
    --controller-lr-schedule cosine \
    --out-dir "$out_dir"
  echo "STAGE $label COMPLETE $(date --iso-8601=seconds)"
}

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CHECKPOINT=$checkpoint"
train_controller strict_id10to20 20
train_controller extension20to40 40
echo "COMPLETE $(date --iso-8601=seconds)"
