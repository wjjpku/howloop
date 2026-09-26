#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set one physical GPU}"
if (( $# == 0 )); then
  echo "usage: $0 SHARED_RANK:STAGE_RANK [...]" >&2
  exit 2
fi

RUNNER=/data/paperexperiment/LooPlus/scripts/run_graph_path_fixed_h1_r48_s16_remote_20260801.sh
LOG_ROOT=/data/paperexperiment/logs/graph_path_fixed_h1_j_20260801
mkdir -p "$LOG_ROOT"

for specification in "$@"; do
  if [[ ! "$specification" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "invalid rank specification: $specification" >&2
    exit 2
  fi
  shared_rank=${specification%%:*}
  stage_rank=${specification##*:}
  label=r${shared_rank}_s${stage_rank}
  echo "$(date -Is) START fixed-H1 identity ${label} gpu=${CUDA_VISIBLE_DEVICES}"
  RUN_VARIANT=pure_identity \
  SINGLE_INITIALIZATION=identity \
  SINGLE_INIT_ARTIFACT= \
  SHARED_RANK="$shared_rank" \
  STAGE_RANK="$stage_rank" \
    "$RUNNER" \
    >>"$LOG_ROOT/grid_driver_${label}_gpu${CUDA_VISIBLE_DEVICES}.log" 2>&1
  echo "$(date -Is) COMPLETE fixed-H1 identity ${label} gpu=${CUDA_VISIBLE_DEVICES}"
done
