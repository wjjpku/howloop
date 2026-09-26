#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 GPU MAX_DEPTH LOOPS SEED OUT_DIR LOG_PATH" >&2
  exit 2
fi

gpu="$1"
max_depth="$2"
loops="$3"
seed="$4"
out_dir="$5"
log_path="$6"

mkdir -p "${out_dir}" "$(dirname "${log_path}")"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec >"${log_path}" 2>&1

echo "START $(date --iso-8601=seconds)"
/data/paperexperiment/.venvs/loopreasoner/bin/python -u -m reasoning_loop.graph_path_loop \
  --node-count 8 \
  --max-depth "${max_depth}" \
  --d-model 256 \
  --n-heads 4 \
  --d-mlp 1024 \
  --n-layers 2 \
  --loops "${loops}" \
  --steps 20000 \
  --batch-size 512 \
  --eval-batch-size 1024 \
  --eval-batches 16 \
  --eval-every 1000 \
  --print-every 1000 \
  --lr 0.0003 \
  --weight-decay 0.3 \
  --warmup-steps 500 \
  --grad-clip 1.0 \
  --seed "${seed}" \
  --dropout 0.0 \
  --aux-loss 0.0 \
  --device cuda \
  --amp \
  --no-compile \
  --save-checkpoints \
  --out-dir "${out_dir}"
echo "COMPLETE $(date --iso-8601=seconds)"
