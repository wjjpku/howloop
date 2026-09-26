#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_INDEX ARM:SEED [ARM:SEED ...]" >&2
  exit 2
fi

gpu_index=$1
shift
repo_root=${TYPED_RELATION_REPO_ROOT:-/data/paperexperiment/LooPlus}
out_root=${TYPED_RELATION_OUT_ROOT:-/data/paperexperiment/typed_relation_composition_20260717}
log_root=${TYPED_RELATION_LOG_ROOT:-/data/paperexperiment/logs/typed_relation_composition_20260717}
python_bin=${TYPED_RELATION_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}

mkdir -p "$out_root" "$out_root/analysis" "$log_root"
cd "$repo_root"

for sample in 1 2; do
  process_rows=$(
    nvidia-smi -i "$gpu_index" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  if [[ -n "$process_rows" ]]; then
    echo "physical GPU $gpu_index has an existing compute process" >&2
    exit 1
  fi
  if (( sample == 1 )); then
    sleep 60
  fi
done

for spec in "$@"; do
  if [[ "$spec" != *:* ]]; then
    echo "invalid run spec: $spec" >&2
    exit 2
  fi
  arm=${spec%%:*}
  seed=${spec#*:}
  architecture=looped
  configured_loops=3
  train_loops=2
  objective=final
  anchor_weight=0
  analyze_circuit=0
  case "$arm" in
    l1_final)
      train_loops=1
      ;;
    shared2_final)
      ;;
    anchor005)
      objective=anchor
      anchor_weight=0.05
      ;;
    anchor010)
      objective=anchor
      anchor_weight=0.10
      analyze_circuit=1
      ;;
    shared2_staged)
      objective=staged
      anchor_weight=1.0
      analyze_circuit=1
      ;;
    unshared2_staged)
      architecture=unshared
      configured_loops=2
      objective=staged
      anchor_weight=1.0
      ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 2
      ;;
  esac

  run_name="typed_${arm}_N8_d32_seed${seed}"
  run_dir="$out_root/$run_name"
  if [[ ! -f "$run_dir/summary.json" ]]; then
    echo "training $run_name on physical GPU $gpu_index"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH="$repo_root" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m reasoning_loop.typed_relation_train \
        --node-count 8 \
        --architecture "$architecture" \
        --d-model 32 \
        --n-heads 4 \
        --d-mlp 128 \
        --configured-loops "$configured_loops" \
        --train-loops "$train_loops" \
        --objective "$objective" \
        --anchor-weight "$anchor_weight" \
        --train-visibility full \
        --steps 5000 \
        --batch-size 512 \
        --eval-batch-size 2048 \
        --eval-batches 4 \
        --final-eval-batches 16 \
        --eval-every 250 \
        --print-every 250 \
        --transplant-examples 8192 \
        --lr 1e-3 \
        --seed "$seed" \
        --device cuda \
        --run-name "$run_name" \
        --out-dir "$out_root" \
        >"$log_root/${run_name}.log" 2>&1
  fi

  if (( analyze_circuit )); then
    analysis_dir="$out_root/analysis/$run_name"
    if [[ ! -f "$analysis_dir/summary.json" ]]; then
      echo "analyzing $run_name on physical GPU $gpu_index"
      CUDA_VISIBLE_DEVICES="$gpu_index" \
        PYTHONPATH="$repo_root" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$python_bin" -m reasoning_loop.typed_relation_circuit \
          --checkpoint "$run_dir/final.pt" \
          --out-dir "$analysis_dir" \
          --examples 4096 \
          --device cuda \
          >>"$log_root/${run_name}.log" 2>&1
    fi
  fi
done
