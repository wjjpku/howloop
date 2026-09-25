#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

gpu_index=$1
repo_root=${SHARDED_REUSE_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${SHARDED_REUSE_OUT_ROOT:-/data/wujiaju/sharded_evidence_reuse_20260715}
log_root=${SHARDED_REUSE_LOG_ROOT:-/data/wujiaju/logs/sharded_evidence_reuse_20260715}
python_bin=${SHARDED_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
mkdir -p "$out_root/branches" "$log_root"

run_job() {
  local target_condition=$1
  local seed=$2
  local run_name=$3
  local checkpoint=$4
  local job_root="$out_root/branches/$run_name"
  mkdir -p "$job_root"
  echo "starting $run_name from $checkpoint"
  (
    cd "$repo_root"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.sharded_evidence_train \
        --conditions "$target_condition" \
        --seeds "$seed" \
        --evidence-count 5 \
        --d-model 64 \
        --n-heads 4 \
        --d-mlp 256 \
        --loops 5 \
        --steps 1000 \
        --stop-step 1000 \
        --checkpoint-every 500 \
        --batch-size 256 \
        --eval-batch-size 512 \
        --eval-batches 4 \
        --eval-every 100 \
        --print-every 100 \
        --warmup-steps 50 \
        --device cuda \
        --run-name "$run_name" \
        --resume "$checkpoint" \
        --out-dir "$job_root"
  ) >"$log_root/$run_name.log" 2>&1
  echo "finished $run_name"
}

for seed in 0 1 2; do
  full_checkpoint="$out_root/full_d64_L5_seed${seed}/checkpoint_step_0000500.pt"
  shard_checkpoint="$out_root/shard_d64_L5_seed${seed}/checkpoint_step_0000500.pt"
  if [[ ! -f "$full_checkpoint" || ! -f "$shard_checkpoint" ]]; then
    echo "missing base checkpoint for seed $seed" >&2
    exit 1
  fi
  run_job full "$seed" "continue_full_seed${seed}" "$full_checkpoint"
  run_job shard "$seed" "full_to_shard_seed${seed}" "$full_checkpoint"
  run_job shard "$seed" "continue_shard_seed${seed}" "$shard_checkpoint"
  run_job full "$seed" "shard_to_full_seed${seed}" "$shard_checkpoint"
done

touch "$out_root/branch_queue_complete"
echo "completed sharded-evidence branch queue"
