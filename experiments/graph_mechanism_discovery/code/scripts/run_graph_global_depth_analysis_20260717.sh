#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_INDEX fixed|sparse|sparse_staged|coprime_no_d1|even_no_d1|primitive_only|anchor01|anchor02|anchor03|anchor04|anchor05|anchor10|anchor20|dense SEED [SEED ...]" >&2
  exit 2
fi

gpu_index=$1
arm=$2
shift 2
case "$arm" in
  fixed|sparse|sparse_staged|coprime_no_d1|even_no_d1|primitive_only|anchor01|anchor02|anchor03|anchor04|anchor05|anchor10|anchor20|dense) ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

repo_root=${GLOBAL_DEPTH_REPO_ROOT:-/data/wujiaju/LooPlus}
out_root=${GLOBAL_DEPTH_OUT_ROOT:-/data/wujiaju/global_depth_supervision_20260717}
log_root=${GLOBAL_DEPTH_LOG_ROOT:-/data/wujiaju/logs/global_depth_supervision_20260717}
python_bin=${GLOBAL_DEPTH_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}
batch_size=${GLOBAL_DEPTH_ANALYSIS_BATCH_SIZE:-512}
batches=${GLOBAL_DEPTH_ANALYSIS_BATCHES:-32}
run_suffix=${GLOBAL_DEPTH_RUN_SUFFIX:-}

mkdir -p "$out_root/analysis" "$log_root"
cd "$repo_root"

process_rows=$(
  nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
)
if [[ -n "$process_rows" ]]; then
  echo "physical GPU $gpu_index has an existing compute process; refusing to share" >&2
  exit 1
fi

for seed in "$@"; do
  run_name="variable_${arm}_N16_D8_d128_seed${seed}${run_suffix}"
  checkpoint="$out_root/variable_unroll/$run_name/final.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "analyzing $run_name on physical GPU $gpu_index"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONPATH="$repo_root" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m reasoning_loop.graph_path_temporal_intervention \
      --run "$run_name=$checkpoint" \
      --out-dir "$out_root/analysis" \
      --donor-loops 1 2 3 4 5 6 7 8 \
      --receiver-loops 1 2 3 4 5 6 7 8 \
      --max-delta 3 \
      --base-loops 8 \
      --max-path-position 12 \
      --batch-size "$batch_size" \
      --batches "$batches" \
      --device cuda \
      >"$log_root/${run_name}_analysis.log" 2>&1
done
