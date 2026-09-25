#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX fixed|sparse|sparse_staged|coprime_no_d1|even_no_d1|primitive_only|anchor01|anchor02|anchor03|anchor04|anchor05|anchor10|anchor20|dense|depth_token_one_step SEED [SEED ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
shift 2
case "$arm" in
  fixed|sparse|sparse_staged|coprime_no_d1|even_no_d1|primitive_only|anchor01|anchor02|anchor03|anchor04|anchor05|anchor10|anchor20|dense|depth_token_one_step) ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${GLOBAL_DEPTH_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${GLOBAL_DEPTH_OUT_ROOT:-/data/wujiaju/global_depth_supervision_20260717}
log_root=${GLOBAL_DEPTH_LOG_ROOT:-/data/wujiaju/logs/global_depth_supervision_20260717}
python_bin=${GLOBAL_DEPTH_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
steps=${GLOBAL_DEPTH_STEPS:-20000}
batch_size=${GLOBAL_DEPTH_BATCH_SIZE:-512}
eval_batch_size=${GLOBAL_DEPTH_EVAL_BATCH_SIZE:-2048}
eval_batches=${GLOBAL_DEPTH_EVAL_BATCHES:-32}
eval_every=${GLOBAL_DEPTH_EVAL_EVERY:-1000}
print_every=${GLOBAL_DEPTH_PRINT_EVERY:-1000}
save_eval_checkpoints=${GLOBAL_DEPTH_SAVE_EVAL_CHECKPOINTS:-0}
init_checkpoint=${GLOBAL_DEPTH_INIT_CHECKPOINT:-}
run_suffix=${GLOBAL_DEPTH_RUN_SUFFIX:-}
declared_peak_mib=4096
reserve_mib=16384
minimum_free_mib=$((declared_peak_mib + reserve_mib))
minimum_disk_mib=6144
if [[ "$save_eval_checkpoints" == "1" ]]; then
  save_eval_flag=(--save-eval-checkpoints)
else
  save_eval_flag=()
fi
if [[ -n "$init_checkpoint" ]]; then
  init_flag=(--init-checkpoint "$init_checkpoint")
else
  init_flag=()
fi

mkdir -p "$out_root/runtime_manifests" "$log_root"
cd "$repo_root"

check_launch_gate() {
  local run_name=$1
  local manifest="$out_root/runtime_manifests/${run_name}.txt"
  local disk_available_mib
  disk_available_mib=$(df -Pm /data | awk 'NR==2 {print $4}')
  if (( disk_available_mib < minimum_disk_mib )); then
    echo "insufficient /data space: ${disk_available_mib} MiB" >&2
    exit 1
  fi
  {
    echo "run_name=$run_name"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "physical_gpu=$gpu_index"
    echo "declared_peak_mib=$declared_peak_mib"
    echo "reserve_mib=$reserve_mib"
    echo "disk_available_mib=$disk_available_mib"
    echo "shared_gpu=false"
  } >"$manifest"

  for sample in 1 2; do
    local free_mib utilization process_rows
    IFS=, read -r free_mib utilization < <(
      nvidia-smi -i "$gpu_index" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        tr -d ' '
    )
    process_rows=$(
      nvidia-smi -i "$gpu_index" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
    )
    {
      echo "sample_${sample}_timestamp=$(date --iso-8601=seconds)"
      echo "sample_${sample}_free_mib=$free_mib"
      echo "sample_${sample}_utilization=$utilization"
      echo "sample_${sample}_processes=${process_rows:-none}"
    } >>"$manifest"
    if [[ -n "$process_rows" ]]; then
      echo "physical GPU $gpu_index has an existing compute process; refusing to share" >&2
      exit 1
    fi
    if (( free_mib < minimum_free_mib || utilization > 10 )); then
      echo "physical GPU $gpu_index failed launch gate" >&2
      exit 1
    fi
    if (( sample == 1 )); then
      sleep 60
    fi
  done
}

start_gpu_monitor() {
  local run_name=$1
  gpu_monitor_file="$out_root/runtime_manifests/${run_name}_gpu_samples.csv"
  echo "timestamp,memory_used_mib,utilization" >"$gpu_monitor_file"
  (
    while true; do
      local used_mib utilization
      IFS=, read -r used_mib utilization < <(
        nvidia-smi -i "$gpu_index" \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits |
          tr -d ' '
      )
      echo "$(date --iso-8601=seconds),$used_mib,$utilization"
      sleep 1
    done
  ) >>"$gpu_monitor_file" &
  gpu_monitor_pid=$!
}

stop_gpu_monitor() {
  local run_name=$1
  kill "$gpu_monitor_pid" 2>/dev/null || true
  wait "$gpu_monitor_pid" 2>/dev/null || true
  local observed_peak_mib
  observed_peak_mib=$(
    awk -F, 'NR > 1 && $2 + 0 > peak {peak=$2 + 0} END {print peak + 0}' \
      "$gpu_monitor_file"
  )
  echo "observed_peak_mib=$observed_peak_mib" \
    >>"$out_root/runtime_manifests/${run_name}.txt"
  if (( observed_peak_mib > declared_peak_mib )); then
    echo "observed peak ${observed_peak_mib} MiB exceeded declared peak" >&2
    exit 1
  fi
}

for seed in "$@"; do
  if [[ "$arm" == "depth_token_one_step" ]]; then
    run_name="depth_token_one_step_N16_D8_d128_seed${seed}${run_suffix}"
    check_launch_gate "$run_name"
    echo "training $run_name on physical GPU $gpu_index"
    start_gpu_monitor "$run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m reasoning_loop.graph_path_loop \
        --node-count 16 \
        --max-depth 8 \
        --d-model 128 \
        --n-heads 4 \
        --d-mlp 512 \
        --n-layers 2 \
        --loops 1 \
        --steps "$steps" \
        --batch-size "$batch_size" \
        --eval-batch-size "$eval_batch_size" \
        --eval-batches "$eval_batches" \
        --eval-every "$eval_every" \
        --print-every "$print_every" \
        --warmup-steps 500 \
        --seed "$seed" \
        --device cuda \
        --out-dir "$out_root/depth_token_one_step/seed${seed}" \
        >"$log_root/${run_name}.log" 2>&1
    run_status=$?
    set -e
    stop_gpu_monitor "$run_name"
    if (( run_status != 0 )); then
      exit "$run_status"
    fi
    continue
  fi

  block_schedule=all_blocks
  case "$arm" in
    fixed)
      train_depths=(8)
      interpolation_depths=()
      train_depth_weights=()
      ;;
    sparse)
      train_depths=(1 2 4 6 8)
      interpolation_depths=(3 5 7)
      train_depth_weights=()
      ;;
    sparse_staged)
      train_depths=(1 2 4 6 8)
      interpolation_depths=(3 5 7)
      train_depth_weights=()
      block_schedule=first_block_once
      ;;
    coprime_no_d1)
      train_depths=(2 3 5 8)
      interpolation_depths=(1 4 6 7)
      train_depth_weights=()
      ;;
    even_no_d1)
      train_depths=(2 4 6 8)
      interpolation_depths=(1 3 5 7)
      train_depth_weights=()
      ;;
    primitive_only)
      train_depths=(1)
      interpolation_depths=(2 3 4 5 6 7 8)
      train_depth_weights=()
      ;;
    anchor01)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.01 0.2475 0.2475 0.2475 0.2475)
      ;;
    anchor02)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.02 0.245 0.245 0.245 0.245)
      ;;
    anchor03)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.03 0.2425 0.2425 0.2425 0.2425)
      ;;
    anchor04)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.04 0.24 0.24 0.24 0.24)
      ;;
    anchor05)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.05 0.2375 0.2375 0.2375 0.2375)
      ;;
    anchor10)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.10 0.225 0.225 0.225 0.225)
      ;;
    anchor20)
      train_depths=(1 2 3 5 8)
      interpolation_depths=(4 6 7)
      train_depth_weights=(0.20 0.20 0.20 0.20 0.20)
      ;;
    dense)
      train_depths=(1 2 3 4 5 6 7 8)
      interpolation_depths=()
      train_depth_weights=()
      ;;
  esac
  if (( ${#train_depth_weights[@]} )); then
    train_depth_weight_flag=(--train-depth-weights "${train_depth_weights[@]}")
  else
    train_depth_weight_flag=()
  fi
  run_name="variable_${arm}_N16_D8_d128_seed${seed}${run_suffix}"
  check_launch_gate "$run_name"
  echo "training $run_name on physical GPU $gpu_index"
  start_gpu_monitor "$run_name"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m reasoning_loop.graph_path_variable_unroll \
      --node-count 16 \
      --max-train-depth 8 \
      --max-eval-depth 12 \
      --train-depths "${train_depths[@]}" \
      "${train_depth_weight_flag[@]}" \
      --interpolation-depths "${interpolation_depths[@]}" \
      --extrapolation-depths 9 10 11 12 \
      --d-model 128 \
      --n-heads 4 \
      --d-mlp 512 \
      --n-layers 2 \
      --block-schedule "$block_schedule" \
      --steps "$steps" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --eval-batches "$eval_batches" \
      --eval-every "$eval_every" \
      --print-every "$print_every" \
      --warmup-steps 500 \
      --seed "$seed" \
      --device cuda \
      --arm-name "$arm" \
      --run-name "$run_name" \
      --out-dir "$out_root/variable_unroll" \
      "${save_eval_flag[@]}" \
      "${init_flag[@]}" \
      >"$log_root/${run_name}.log" 2>&1
  run_status=$?
  set -e
  stop_gpu_monitor "$run_name"
  if (( run_status != 0 )); then
    exit "$run_status"
  fi
done
