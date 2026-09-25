#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 SHARED_RANK:STAGE_RANK [...]" >&2
  exit 2
fi

PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
REPO_DIR=/data/wujiaju/LooPlus
TRAIN_MODULE=reasoning_loop.train_graph_path_age_specific_j_bank
CHECKPOINT=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
FULL_BANK=/data/wujiaju/graph_path_age_specific_j_bank_20260801/seed0_full_affine_balanced5_continuation/age_specific_j_bank_after_B74_26_lr1e6.pt
RESULT_ROOT=/data/wujiaju/graph_path_shared_stage_j_rank_grid_20260801
LOG_ROOT=/data/wujiaju/logs/graph_path_shared_stage_j_rank_grid_20260801

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
cd "$REPO_DIR"

validate_result() {
  local summary_path=$1
  local artifact_path=$2
  "$PYTHON_BIN" - "$summary_path" "$artifact_path" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
artifact_path = pathlib.Path(sys.argv[2])
payload = json.loads(summary_path.read_text())
if payload.get("status") != "complete":
    raise SystemExit(f"incomplete summary: {summary_path}")
if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
    raise SystemExit(f"missing artifact: {artifact_path}")
print(f"validated {summary_path} and {artifact_path}", flush=True)
PY
}

run_one() {
  local shared_rank=$1
  local stage_rank=$2
  local label="r${shared_rank}_s${stage_rank}"
  local single_dir="$RESULT_ROOT/single_${label}"
  local focused_dir="$RESULT_ROOT/focused5_${label}"
  local single_artifact="$single_dir/age_specific_j_bank_after_T01_R1.pt"
  local focused_artifact="$focused_dir/age_specific_j_bank_after_T05_R5.pt"

  echo "[$(date -Is)] START ${label} physical_gpu=${CUDA_VISIBLE_DEVICES}" >&2

  if [[ -s "$single_dir/summary.json" && -s "$single_artifact" ]]; then
    validate_result "$single_dir/summary.json" "$single_artifact"
    echo "[$(date -Is)] REUSE single_${label}" >&2
  else
    mkdir -p "$single_dir"
    "$PYTHON_BIN" -u -m "$TRAIN_MODULE" \
      --checkpoint "$CHECKPOINT" \
      --phase-summary "$PHASE_SUMMARY" \
      --out-dir "$single_dir" \
      --rank "$shared_rank" \
      --stage-rank "$stage_rank" \
      --map-architecture shared_diagonal_stage_lora \
      --initialization age_specific_bank \
      --bank-init-artifact "$FULL_BANK" \
      --curriculum single_back \
      --rollback-composition product \
      --warmup-fraction 0.1 \
      --warmup-start-factor 0.1 \
      --diagonal-lr-multiplier 0.1 \
      --seed 820001 \
      >"$LOG_ROOT/single_${label}.log" 2>&1
    validate_result "$single_dir/summary.json" "$single_artifact"
  fi

  if [[ -s "$focused_dir/summary.json" && -s "$focused_artifact" ]]; then
    validate_result "$focused_dir/summary.json" "$focused_artifact"
    echo "[$(date -Is)] REUSE focused5_${label}" >&2
  else
    mkdir -p "$focused_dir"
    "$PYTHON_BIN" -u -m "$TRAIN_MODULE" \
      --checkpoint "$CHECKPOINT" \
      --phase-summary "$PHASE_SUMMARY" \
      --out-dir "$focused_dir" \
      --rank "$shared_rank" \
      --stage-rank "$stage_rank" \
      --map-architecture shared_diagonal_stage_lora \
      --initialization age_specific_bank \
      --bank-init-artifact "$single_artifact" \
      --curriculum focused5 \
      --rollback-composition product \
      --warmup-fraction 0.1 \
      --warmup-start-factor 0.1 \
      --diagonal-lr-multiplier 0.1 \
      --seed 820001 \
      >"$LOG_ROOT/focused5_${label}.log" 2>&1
    validate_result "$focused_dir/summary.json" "$focused_artifact"
  fi

  echo "[$(date -Is)] COMPLETE ${label} physical_gpu=${CUDA_VISIBLE_DEVICES}" >&2
}

for specification in "$@"; do
  if [[ ! "$specification" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "invalid rank specification: $specification" >&2
    exit 2
  fi
  run_one "${specification%%:*}" "${specification##*:}"
done
