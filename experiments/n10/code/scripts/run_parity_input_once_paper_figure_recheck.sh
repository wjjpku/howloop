#!/usr/bin/env bash
set -euo pipefail

run_root=/data/paperexperiment/parity_input_once_20260811
seed="${PARITY_SEED:?set PARITY_SEED}"
code_root="$run_root/evaluation_code"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/best.pt"
strict_controller="$run_root/controllers/seed${seed}/strict_id10to20/controller.pt"
extension_controller="$run_root/controllers/seed${seed}/extension20to40/controller.pt"
out_root="$run_root/paper_figure_recheck/seed${seed}"
log_file="/data/paperexperiment/logs/parity_input_once_20260811/paper_figure_recheck_seed${seed}.log"
pid_file="$run_root/paper_figure_recheck_seed${seed}.pid"

mkdir -p "$out_root" "$(dirname "$log_file")"
echo $$ > "$pid_file"
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root"

run_stage() {
  local name="$1"
  shift
  local done_file="$out_root/.${name}.complete"
  if [[ -f "$done_file" ]]; then
    echo "STAGE $name ALREADY_COMPLETE $(date --iso-8601=seconds)"
    return
  fi
  echo "STAGE $name START $(date --iso-8601=seconds)"
  "$@"
  touch "$done_file"
  echo "STAGE $name COMPLETE $(date --iso-8601=seconds)"
}

far_lengths=(10 20 24 32 40 48 64 80 100 128 160 200)
long_lengths=()
for ((n=100; n<=300; n+=5)); do long_lengths+=("$n"); done
for ((n=310; n<=600; n+=10)); do long_lengths+=("$n"); done
for ((n=620; n<=1000; n+=20)); do long_lengths+=("$n"); done

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CHECKPOINT=$checkpoint"

for label in strict_id10to20 extension20to40; do
  if [[ "$label" == strict_id10to20 ]]; then
    controller="$strict_controller"
  else
    controller="$extension_controller"
  fi

  run_stage "${label}_far_horizon" \
    "$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
      --task parity \
      --checkpoint "$checkpoint" \
      --controller "$controller" \
      --lengths "${far_lengths[@]}" \
      --examples 512 \
      --max-batch-size 32 \
      --token-budget 8192 \
      --seed "$((2026083101 + 100 * seed))" \
      --device cuda \
      --out-dir "$out_root/$label/far_horizon"

  run_stage "${label}_heatmap" \
    "$python_bin" -u "$code_root/scripts/evaluate_parity_loop_depth_heatmap.py" \
      --checkpoint "$checkpoint" \
      --controller "$controller" \
      --out-dir "$out_root/$label/loop_depth_heatmap" \
      --min-length 1 \
      --max-length 100 \
      --max-loop 112 \
      --examples 256 \
      --batch-size 64 \
      --seed "$((2026083201 + 100 * seed))" \
      --device cuda

  run_stage "${label}_four_phase" \
    "$python_bin" -u "$code_root/reasoning_loop/analyze_parity_four_phase.py" \
      --checkpoint "$checkpoint" \
      --controller "$controller" \
      --out-dir "$out_root/$label/four_phase" \
      --device cuda \
      --batch-size 256 \
      --batches 2 \
      --causal-batch-size 256 \
      --discovery-seed "$((2026083301 + 100 * seed))" \
      --evaluation-seed "$((2026083401 + 100 * seed))" \
      --causal-seed "$((2026083501 + 100 * seed))"
done

run_stage extension20to40_j_svd \
  "$python_bin" -u "$code_root/reasoning_loop/evaluate_parity_j_svd_interventions.py" \
    --checkpoint "$checkpoint" \
    --controller "$extension_controller" \
    --out-dir "$out_root/extension20to40/j_svd" \
    --device cuda \
    --lengths 20 40 50 75 100 \
    --batch-size 32 \
    --batches 4 \
    --post-target-steps 8 \
    --seed "$((2026083601 + 100 * seed))"

run_stage raw_circuit \
  "$python_bin" -u "$code_root/reasoning_loop/analyze_parity_circuit.py" \
    --checkpoint "$checkpoint" \
    --output-dir "$out_root/raw_circuit" \
    --device cuda \
    --batch-size 512 \
    --seed "$((2026083701 + 100 * seed))"

run_stage extension20to40_long_horizon \
  "$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
    --task parity \
    --checkpoint "$checkpoint" \
    --controller "$extension_controller" \
    --lengths "${long_lengths[@]}" \
    --examples 32 \
    --max-batch-size 8 \
    --token-budget 8192 \
    --seed "$((2026083801 + 100 * seed))" \
    --device cuda \
    --out-dir "$out_root/extension20to40/long_horizon"

run_stage extension20to40_diagonal_band \
  "$python_bin" -u "$code_root/scripts/evaluate_parity_diagonal_band.py" \
    --checkpoint "$checkpoint" \
    --controller "$extension_controller" \
    --out-dir "$out_root/extension20to40/diagonal_band" \
    --min-length 1 \
    --max-length 500 \
    --length-step 1 \
    --max-loop 510 \
    --half-width 10 \
    --examples 64 \
    --batch-size 32 \
    --seed "$((2026083901 + 100 * seed))" \
    --device cuda

echo "COMPLETE $(date --iso-8601=seconds)"
