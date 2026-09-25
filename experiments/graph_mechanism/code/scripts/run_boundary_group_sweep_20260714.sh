#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wujiaju/LooPlus
PY=/data/wujiaju/.venvs/loopreasoner/bin/python
LOG_ROOT=/data/wujiaju/logs/post_convergence_boundary_group_sweep_20260714

mkdir -p "$LOG_ROOT"
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "START $(date --iso-8601=seconds)"

pids=()
labels=()
for groups in 2 4; do
  output_dir="/data/wujiaju/post_convergence_boundary_g${groups}_20260714"
  mkdir -p "$output_dir"
  for seed in 0 1 2 3 4 5; do
    gpu=$((2 + seed % 3))
    log="$LOG_ROOT/g${groups}_seed${seed}.log"
    echo "LAUNCH G${groups} seed=${seed} gpu=${gpu} output=${output_dir}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m small_modadd.post_convergence_overloop \
      --seeds "$seed" \
      --output-dir "$output_dir" \
      --device cuda \
      --p 23 \
      --d-model 64 \
      --n-heads 4 \
      --d-mlp 128 \
      --training-loops 6 \
      --batch-size 128 \
      --lr 0.003 \
      --weight-decay 0.0001 \
      --eval-every 10 \
      --interval 2000 \
      --max-offset 20000 \
      --max-pre-hit 20000 \
      --max-loops 100 \
      --norm-type rmsnorm \
      --loop-boundary-norm \
      --loop-boundary-groups "$groups" \
      --no-merge >"$log" 2>&1 &
    pids+=("$!")
    labels+=("G${groups}/seed${seed}")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "TRAIN_COMPLETE ${labels[$index]}"
  else
    echo "TRAIN_FAILED ${labels[$index]} log=$LOG_ROOT/${labels[$index]//\//_}.log"
    failed=1
  fi
done
if (( failed )); then
  echo "ABORT: at least one training process failed"
  exit 1
fi

for groups in 2 4; do
  output_dir="/data/wujiaju/post_convergence_boundary_g${groups}_20260714"
  "$PY" -m small_modadd.post_convergence_overloop \
    --output-dir "$output_dir" \
    --merge-only >"$LOG_ROOT/g${groups}_merge.log" 2>&1
done

residual_pids=()
for groups in 2 4; do
  experiment_dir="/data/wujiaju/post_convergence_boundary_g${groups}_20260714"
  residual_dir="/data/wujiaju/post_convergence_boundary_g${groups}_residual_20260714"
  mapfile -t complete_seeds < <(
    for payload in "$experiment_dir"/seed_*.json; do
      "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["seed"] if p["status"] == "complete" else "")' "$payload"
    done | awk 'NF'
  )
  if (( ${#complete_seeds[@]} == 0 )); then
    echo "RESIDUAL_SKIP G${groups}: no completed seeds"
    continue
  fi
  gpu=$((groups == 2 ? 2 : 3))
  echo "RESIDUAL_LAUNCH G${groups} gpu=${gpu} seeds=${complete_seeds[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m small_modadd.post_convergence_residual_dynamics \
    --experiment-dir "$experiment_dir" \
    --output-dir "$residual_dir" \
    --seeds "${complete_seeds[@]}" \
    --max-loops 100 \
    --device cuda >"$LOG_ROOT/g${groups}_residual.log" 2>&1 &
  residual_pids+=("$!")
done

for pid in "${residual_pids[@]}"; do
  wait "$pid"
done

echo "COMPLETE $(date --iso-8601=seconds)"
