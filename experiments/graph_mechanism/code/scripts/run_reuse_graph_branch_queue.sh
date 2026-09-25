#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 final|transition|portable GPU_INDEX" >&2
  exit 2
fi

source_mode=$1
gpu_index=$2
case "$source_mode" in
  final|transition|portable) ;;
  *)
    echo "unknown source mode: $source_mode" >&2
    exit 2
    ;;
esac
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  echo "GPU_INDEX must be a nonnegative integer" >&2
  exit 2
fi

repo_root=${LOOP_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${LOOP_REUSE_OUT_ROOT:-/data/wujiaju/loop_reuse_graph_pilot_20260715}
log_root=${LOOP_REUSE_LOG_ROOT:-/data/wujiaju/logs/reuse_graph_pilot_20260715}
python_bin=${LOOP_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
mkdir -p "$out_root/jobs" "$log_root"

find_base_checkpoint() {
  local mode=$1
  local seed=$2
  local -a matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(
    find "$out_root" -type f \
      -path "*/base_${mode}_seed${seed}/checkpoint_step_0005000.pt" \
      -print | sort
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one base checkpoint for mode=$mode seed=$seed, found ${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

run_job() {
  local target_mode=$1
  local seed=$2
  local run_name=$3
  local checkpoint=$4
  local job_root="$out_root/jobs/$run_name"
  local log_path="$log_root/$run_name.log"
  mkdir -p "$job_root"
  echo "starting $run_name on physical GPU $gpu_index from $checkpoint"
  (
    cd "$repo_root"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.portable_component_train \
        --modes "$target_mode" \
        --seeds "$seed" \
        --node-count 8 \
        --max-depth 6 \
        --d-model 64 \
        --n-heads 4 \
        --d-mlp 256 \
        --n-layers 2 \
        --loops 6 \
        --steps 10000 \
        --stop-step 10000 \
        --checkpoint-every 5000 \
        --batch-size 128 \
        --eval-batch-size 256 \
        --eval-batches 4 \
        --overloop 10 \
        --eval-every 500 \
        --print-every 500 \
        --device cuda \
        --run-name "$run_name" \
        --resume "$checkpoint" \
        --out-dir "$job_root"
  ) >"$log_path" 2>&1
  echo "finished $run_name"
}

for seed in 0 1 2; do
  checkpoint=$(find_base_checkpoint "$source_mode" "$seed")
  run_job "$source_mode" "$seed" "continue_${source_mode}_seed${seed}" "$checkpoint"
  case "$source_mode" in
    final)
      run_job transition "$seed" "final_to_transition_seed${seed}" "$checkpoint"
      run_job portable "$seed" "final_to_portable_seed${seed}" "$checkpoint"
      ;;
    transition)
      run_job final "$seed" "transition_to_final_seed${seed}" "$checkpoint"
      ;;
    portable)
      run_job final "$seed" "portable_to_final_seed${seed}" "$checkpoint"
      ;;
  esac
done

touch "$out_root/queue_${source_mode}_complete"
echo "completed source queue: $source_mode"
