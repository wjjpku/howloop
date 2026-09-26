#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one physical GPU}"

PYTHON_BIN=/data/paperexperiment/.venvs/loopreasoner/bin/python
REPO_DIR=/data/paperexperiment/LooPlus
TRAIN_MODULE=reasoning_loop.train_graph_path_age_specific_j_bank
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE_SUMMARY=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
FULL_BANK=/data/paperexperiment/graph_path_age_specific_j_bank_20260801/seed0_full_affine_balanced5_continuation/age_specific_j_bank_after_B74_26_lr1e6.pt
RUN_VARIANT=${RUN_VARIANT:-paired_fullbank_init}
SINGLE_INITIALIZATION=${SINGLE_INITIALIZATION:-age_specific_bank}
SINGLE_INIT_ARTIFACT=${SINGLE_INIT_ARTIFACT-$FULL_BANK}
SHARED_RANK=${SHARED_RANK:-48}
STAGE_RANK=${STAGE_RANK:-16}
RANK_LABEL=r${SHARED_RANK}_s${STAGE_RANK}
RESULT_ROOT=/data/paperexperiment/graph_path_fixed_h1_j_20260801/${RUN_VARIANT}/${RANK_LABEL}
LOG_ROOT=/data/paperexperiment/logs/graph_path_fixed_h1_j_20260801

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
cd "$REPO_DIR"

run_stage() {
  local label=$1
  local curriculum=$2
  local initialization=$3
  local init_artifact=$4
  local output_dir="$RESULT_ROOT/$label"
  local log="$LOG_ROOT/${RUN_VARIANT}_${RANK_LABEL}_${label}_gpu${CUDA_VISIBLE_DEVICES}.log"
  local final_artifact
  if [[ "$curriculum" == single_back ]]; then
    final_artifact="$output_dir/age_specific_j_bank_after_T01_R1.pt"
  else
    final_artifact="$output_dir/age_specific_j_bank_after_T05_R5.pt"
  fi
  if [[ -s "$output_dir/summary.json" && -s "$final_artifact" ]]; then
    echo "reuse $final_artifact"
    return
  fi
  mkdir -p "$output_dir"
  local init_args=()
  if [[ -n "$init_artifact" ]]; then
    init_args=(--bank-init-artifact "$init_artifact")
  fi
  "$PYTHON_BIN" -u -m "$TRAIN_MODULE" \
    --checkpoint "$CHECKPOINT" \
    --phase-summary "$PHASE_SUMMARY" \
    --out-dir "$output_dir" \
    --rank "$SHARED_RANK" \
    --stage-rank "$STAGE_RANK" \
    --map-architecture shared_diagonal_stage_lora \
    --initialization "$initialization" \
    "${init_args[@]}" \
    --curriculum "$curriculum" \
    --rollback-composition product \
    --fixed-start-age 1 \
    --warmup-fraction 0.1 \
    --warmup-start-factor 0.1 \
    --diagonal-lr-multiplier 0.1 \
    --cuda-memory-fraction 0.02 \
    --seed 820001 \
    >"$log" 2>&1
  "$PYTHON_BIN" - "$output_dir/summary.json" "$final_artifact" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
artifact = pathlib.Path(sys.argv[2])
assert summary["status"] == "complete"
assert summary["training_start_age"] == 1
assert artifact.is_file() and artifact.stat().st_size > 0
print(f"validated {artifact}", flush=True)
PY
}

run_stage single_h1 single_back "$SINGLE_INITIALIZATION" "$SINGLE_INIT_ARTIFACT"
run_stage \
  focused5_h1 \
  focused5 \
  age_specific_bank \
  "$RESULT_ROOT/single_h1/age_specific_j_bank_after_T01_R1.pt"
