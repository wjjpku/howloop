#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_INDEX CODE_ROOT" >&2
  exit 2
fi

GPU_INDEX="$1"
CODE_ROOT="$2"
PYTHON_BIN="/data/wujiaju/.venvs/loopreasoner/bin/python"
OVERLAY_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RESULT_ROOT="/data/wujiaju/parity_computation_mechanism_20260812"
LOG_ROOT="/data/wujiaju/logs"
CHECKPOINT_ROOT="/data/wujiaju/parity_input_once_20260811/backbones"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

for BACKBONE_SEED in 0 1 2; do
  DISCOVERY_SEED=$((2026087101 + BACKBONE_SEED))
  SELECTION_SEED=$((2026087201 + BACKBONE_SEED))
  CHECKPOINT="$CHECKPOINT_ROOT/parity_input_once_seed${BACKBONE_SEED}/best.pt"
  for CAUSAL_PREFIX in 20260873 20260874 20260875; do
    CAUSAL_SEED="${CAUSAL_PREFIX}0$((BACKBONE_SEED + 1))"
    RUN_DIR="$RESULT_ROOT/seed${BACKBONE_SEED}/data${CAUSAL_SEED}"
    LOG_PATH="$LOG_ROOT/parity_computation_stage_b_seed${BACKBONE_SEED}_data${CAUSAL_SEED}.log"
    mkdir -p "$RUN_DIR"
    (
      cd "$OVERLAY_ROOT"
      PYTHONPATH="$OVERLAY_ROOT:$CODE_ROOT" CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" \
        -m reasoning_loop.analyze_parity_computation \
        --checkpoint "$CHECKPOINT" \
        --out-dir "$RUN_DIR" \
        --device cuda \
        --discovery-lengths 12 16 20 24 32 40 \
        --causal-lengths 10 14 18 22 \
        --discovery-batch-size 512 \
        --discovery-batches 2 \
        --causal-batch-size 256 \
        --discovery-seed "$DISCOVERY_SEED" \
        --selection-seed "$SELECTION_SEED" \
        --causal-seed "$CAUSAL_SEED" \
        --random-controls 4 \
        --continuation-calls 2 \
        --backbone-seed "$BACKBONE_SEED"
    ) >"$LOG_PATH" 2>&1
  done
done
