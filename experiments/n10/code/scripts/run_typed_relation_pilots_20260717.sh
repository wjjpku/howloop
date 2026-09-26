#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_INDEX low|high" >&2
  exit 2
fi

gpu_index=$1
suite=$2
case "$suite" in
  low|high) ;;
  *)
    echo "suite must be low or high" >&2
    exit 2
    ;;
esac

repo_root=${TYPED_RELATION_REPO_ROOT:-/data/paperexperiment/LooPlus}
out_root=${TYPED_RELATION_OUT_ROOT:-/data/paperexperiment/typed_relation_composition_20260717}
log_root=${TYPED_RELATION_LOG_ROOT:-/data/paperexperiment/logs/typed_relation_composition_20260717}
python_bin=${TYPED_RELATION_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}

mkdir -p "$out_root" "$log_root"
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

run_one() {
  local arm=$1
  local architecture=$2
  local configured_loops=$3
  local train_loops=$4
  local objective=$5
  local anchor_weight=$6
  local seed=$7
  local run_name="typed_${arm}_N8_d32_seed${seed}"
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
}

if [[ "$suite" == "low" ]]; then
  run_one l1_final looped 3 1 final 0 0
  run_one shared2_final looped 3 2 final 0 0
  run_one anchor0001 looped 3 2 anchor 0.001 0
  run_one anchor0003 looped 3 2 anchor 0.003 0
  run_one anchor001 looped 3 2 anchor 0.01 0
else
  run_one anchor003 looped 3 2 anchor 0.03 0
  run_one anchor005 looped 3 2 anchor 0.05 0
  run_one anchor010 looped 3 2 anchor 0.10 0
  run_one shared2_staged looped 3 2 staged 1.0 0
  run_one unshared2_staged unshared 2 2 staged 1.0 0
fi
