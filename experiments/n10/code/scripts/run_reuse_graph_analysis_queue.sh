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
python_bin=${LOOP_REUSE_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
analysis_root="$out_root/analysis/$source_mode"
mkdir -p "$analysis_root/diagnostics"

declare -a checkpoints=()
while IFS= read -r checkpoint; do
  checkpoints+=("$checkpoint")
done < <(
  find "$out_root" -type f \
    -path "*/base_${source_mode}_seed*/checkpoint_step_0005000.pt" | sort
)
case "$source_mode" in
  final)
    branch_patterns=(continue_final final_to_transition final_to_portable)
    expected=12
    ;;
  transition)
    branch_patterns=(continue_transition transition_to_final)
    expected=9
    ;;
  portable)
    branch_patterns=(continue_portable portable_to_final)
    expected=9
    ;;
esac
for prefix in "${branch_patterns[@]}"; do
  while IFS= read -r checkpoint; do
    checkpoints+=("$checkpoint")
  done < <(
    find "$out_root/jobs" -type f \
      -path "*/${prefix}_seed*/checkpoint_step_0010000.pt" | sort
  )
done
if [[ ${#checkpoints[@]} -ne $expected ]]; then
  echo "expected $expected checkpoints for $source_mode, found ${#checkpoints[@]}" >&2
  printf '%s\n' "${checkpoints[@]}" >&2
  exit 1
fi

declare -a temporal_specs=()
declare -a component_specs=()
for checkpoint in "${checkpoints[@]}"; do
  run_name=$(basename "$(dirname "$checkpoint")")
  echo "diagnosing $run_name"
  (
    cd "$repo_root"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      "$python_bin" -m reasoning_loop.reuse_graph_diagnostics \
        --checkpoint "$checkpoint" \
        --out-dir "$analysis_root/diagnostics/$run_name" \
        --state-loops 1 2 3 4 5 6 \
        --batch-size 128 \
        --batches 4 \
        --max-delta 3 \
        --device cuda
  ) >"$analysis_root/diagnostics/$run_name.log" 2>&1
  temporal_specs+=(--run "$run_name=$checkpoint")
  target_mode=$source_mode
  if [[ "$run_name" == *_to_* ]]; then
    target_mode=${run_name#*_to_}
    target_mode=${target_mode%_seed*}
  elif [[ "$run_name" == continue_* ]]; then
    target_mode=${run_name#continue_}
    target_mode=${target_mode%_seed*}
  fi
  behavior=endpoint
  if [[ "$target_mode" == transition || "$target_mode" == portable ]]; then
    behavior=transition
  fi
  component_specs+=(--run "$run_name:$behavior=$checkpoint")
done

echo "running temporal interventions for $source_mode"
(
  cd "$repo_root"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" -m reasoning_loop.graph_path_temporal_intervention \
      "${temporal_specs[@]}" \
      --out-dir "$analysis_root/temporal" \
      --donor-loops 1 2 3 4 5 6 \
      --receiver-loops 1 2 3 4 5 6 \
      --max-delta 3 \
      --base-loops 6 \
      --max-path-position 8 \
      --batch-size 128 \
      --batches 4 \
      --device cuda
) >"$analysis_root/temporal.log" 2>&1

echo "running component patches for $source_mode"
(
  cd "$repo_root"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    "$python_bin" -m reasoning_loop.graph_path_component_patching \
      "${component_specs[@]}" \
      --out-dir "$analysis_root/components" \
      --state-loops 1 2 3 4 5 6 \
      --batch-size 128 \
      --batches 2 \
      --device cuda
) >"$analysis_root/components.log" 2>&1

touch "$out_root/analysis_${source_mode}_complete"
echo "completed graph analysis queue: $source_mode"
