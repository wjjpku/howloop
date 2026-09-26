#!/bin/bash
set -eu
TASK_GPU=$1
TASK_SHARD=$2
export CUDA_VISIBLE_DEVICES="$TASK_GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME=/data/paperexperiment/cache/huggingface TORCH_HOME=/data/paperexperiment/cache/torch
cd /data/paperexperiment/n10_migration_20260923/code
exec /data/paperexperiment/.venvs/loopreasoner/bin/python -u worker_backbones.py --shard "$TASK_SHARD"
