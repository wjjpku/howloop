#!/bin/bash
set -eu
TASK_ROOT=/data/wujiaju/n10_migration_20260923
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME=/data/wujiaju/cache/huggingface TORCH_HOME=/data/wujiaju/cache/torch
cd "$TASK_ROOT/code"
exec /data/wujiaju/.venvs/loopreasoner/bin/python -u -m reasoning_loop.paper2027_graph_g4_backbone --out-dir "$TASK_ROOT/smoke_L8" --selection-lock "$TASK_ROOT/locks/selection.pt" --final-test-lock "$TASK_ROOT/locks/confirmation.pt" --extra-exclude-locks "$TASK_ROOT/locks/discovery.pt" "$TASK_ROOT/locks/rings.pt" "$TASK_ROOT/locks/smoke.pt" "$TASK_ROOT/locks/donors.pt" --seed 0 --loops 8 --depth-mode uniform --steps 30 --eval-every 10 --batch-size 512 --eval-batch-size 500 --amp
