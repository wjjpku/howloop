#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES="$1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME=/data/wujiaju/cache/huggingface TORCH_HOME=/data/wujiaju/cache/torch
cd /data/wujiaju/n10_migration_20260923/code
exec /data/wujiaju/.venvs/loopreasoner/bin/python -u after_backbones.py --shard "$2"
