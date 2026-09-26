#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PY=/data/paperexperiment/.venvs/loopreasoner/bin/python
REPO=/data/paperexperiment/LooPlus
ROOT=/data/paperexperiment/graph_path_fixed_h1_j_20260802
CHECKPOINT=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
PHASE=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
NATURAL_PROBE=/data/paperexperiment/graph_path_age_probe_adversarial_20260801/seed0_full_affine_balanced5_n1024/adversarial_probe_weights.npz

PARENT=/data/paperexperiment/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16/age_specific_j_bank_after_inverse_reuse_10k.pt
MAIN="$ROOT/path_equivalence_r64_s16_main/age_specific_j_bank_after_equivalent_k3_to_k12.pt"
STAGE16="$ROOT/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k16.pt"
STAGE24="$ROOT/path_equivalence_r64_s16_extension_k24/age_specific_j_bank_after_equivalent_inverse_k3_to_k24.pt"
# Set SELECTED explicitly after the fixed checkpoint-selection evaluation.  The
# newest checkpoint is not assumed to be the best one.
SELECTED="${SELECTED:-$STAGE24}"
SKIP_MULTI="${SKIP_MULTI:-0}"

for required in "$PY" "$CHECKPOINT" "$PHASE" "$NATURAL_PROBE" "$PARENT" "$MAIN" "$STAGE16" "$STAGE24" "$SELECTED"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 1
  fi
done

cd "$REPO"

if [[ "$SKIP_MULTI" != "1" ]]; then
  "$PY" -u -m reasoning_loop.evaluate_graph_path_j_path_equivalence_multiseed \
    --checkpoint "$CHECKPOINT" \
    --bank "parent=$PARENT" \
    --bank "main=$MAIN" \
    --bank "stage16=$STAGE16" \
    --bank "stage24=$STAGE24" \
    --out-dir "$ROOT/eval_multiseed_all_final" \
    --evaluation-seeds 0 1 2 3 4 \
    --examples-per-seed 512 \
    --batch-size 64 \
    --pair-count 7 \
    --back-counts 2 3 4 5 6 7 8 9 10 11 12 16 24 32 \
    --cuda-memory-fraction 0.02
fi

"$PY" -u -m reasoning_loop.analyze_graph_path_j_composition_laws \
  --checkpoint "$CHECKPOINT" \
  --phase-summary "$PHASE" \
  --bank-artifact "$SELECTED" \
  --out-dir "$ROOT/composition_laws_selected" \
  --examples 512 \
  --batch-size 64 \
  --max-steps 7 \
  --cuda-memory-fraction 0.02

"$PY" -u -m reasoning_loop.analyze_graph_path_j_path_equivalence_dynamics \
  --checkpoint "$CHECKPOINT" \
  --phase-summary "$PHASE" \
  --bank-artifacts "$PARENT" "$MAIN" "$STAGE16" "$STAGE24" \
  --labels parent main stage16 stage24 \
  --age-probe "$NATURAL_PROBE" \
  --out-dir "$ROOT/dynamics_parent_main_stage16_final" \
  --examples 768 \
  --batch-size 64 \
  --cuda-memory-fraction 0.02

"$PY" -u -m reasoning_loop.evaluate_graph_path_dynamic_j_schedules_from_tokens \
  --checkpoint "$CHECKPOINT" \
  --bank-artifact "$SELECTED" \
  --out-dir "$ROOT/dynamic_selected_long" \
  --evaluation-seeds 0 1 2 \
  --examples-per-seed 512 \
  --batch-size 128 \
  --max-state 40 \
  --prefix-lengths 1 2 4 7 \
  --cuda-memory-fraction 0.02

"$PY" -u -m reasoning_loop.analyze_graph_path_j_functional_age_probe \
  --checkpoint "$CHECKPOINT" \
  --bank-artifacts "$PARENT" "$MAIN" "$STAGE16" "$STAGE24" \
  --labels parent main stage16 stage24 \
  --natural-age-probe "$NATURAL_PROBE" \
  --out-dir "$ROOT/functional_age_probe_all_final" \
  --examples 1024 \
  --batch-size 64 \
  --cuda-memory-fraction 0.02

"$PY" -u -m reasoning_loop.analyze_graph_path_j_path_equivalence_circuit_drift \
  --checkpoint "$CHECKPOINT" \
  --bank-artifacts "$PARENT" "$MAIN" "$STAGE16" "$STAGE24" \
  --labels parent main stage16 stage24 \
  --out-dir "$ROOT/circuit_drift_all_final" \
  --examples 128 \
  --batch-size 64 \
  --cuda-memory-fraction 0.02

echo "final analysis suite complete"
