#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ACTION PHYSICAL_GPU" >&2
  exit 2
fi

ACTION="$1"
PHYSICAL_GPU="$2"
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be an explicit physical GPU index" >&2
  exit 2
fi

CODE_ROOT="/data/paperexperiment/LooPlus"
OUTPUT_ROOT="/data/paperexperiment/graph_path_telomere_localized_query_D8L8_20260730"
LOG_ROOT="/data/paperexperiment/logs/graph_path_telomere_localized_query_D8L8_20260730"
PYTHON_BIN="/data/paperexperiment/.venvs/loopreasoner/bin/python"
CHECKPOINT="/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt"
PHASE_SUMMARY="/data/paperexperiment/graph_path_telomere_overloop_20260729/formal/D8_L8_seed1/summary.json"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${CODE_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TELOMERE_CUDA_MEMORY_FRACTION="${TELOMERE_CUDA_MEMORY_FRACTION:-0.04}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${CODE_ROOT}"
LOG_PATH="${LOG_ROOT}/${ACTION}_gpu${PHYSICAL_GPU}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

for required in "${CHECKPOINT}" "${PHASE_SUMMARY}"; do
  if [[ ! -r "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 3
  fi
done

write_manifest() {
  local status="$1"
  STATUS="${status}" \
  ACTION_NAME="${ACTION}" \
  PHYSICAL_GPU_VALUE="${PHYSICAL_GPU}" \
  LOG_PATH_VALUE="${LOG_PATH}" \
  OUTPUT_ROOT_VALUE="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": os.environ["STATUS"],
    "action": os.environ["ACTION_NAME"],
    "physical_gpu": int(os.environ["PHYSICAL_GPU_VALUE"]),
    "visible_cuda_device": 0,
    "pid": os.getppid(),
    "log_path": os.environ["LOG_PATH_VALUE"],
    "cuda_memory_fraction": float(
        os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
    ),
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = (
    Path(os.environ["OUTPUT_ROOT_VALUE"])
    / f"{os.environ['ACTION_NAME']}_launcher_manifest.json"
)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

run_one() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local evaluation_batches="$5"
  local age_ladder_batches="$6"
  local extra_loops="$7"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_localized_query \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --evaluation-batch-size 256 \
    --evaluation-batches "${evaluation_batches}" \
    --age-ladder-batches "${age_ladder_batches}" \
    --extra-loops "${extra_loops}" \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --evaluation-seed "$((seed_base + 2))" \
    --ridge 0.01 \
    --device cuda
}

run_dagger() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local dagger_batches="$5"
  local dagger_rounds="$6"
  local evaluation_batches="$7"
  local extra_loops="$8"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_executor_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --dagger-batch-size 256 \
    --dagger-batches "${dagger_batches}" \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-rollout-cycles 4 \
    --evaluation-batch-size 256 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --ranks 32 256 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_position_circuit() {
  local label="$1"
  local seed="$2"
  local batches="$3"
  local extra_loops="$4"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_oracle_position_circuit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 256 \
    --batches "${batches}" \
    --extra-loops "${extra_loops}" \
    --seed "${seed}" \
    --device cuda
}

run_shared_dagger() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local dagger_batches="$5"
  local dagger_rounds="$6"
  local evaluation_batches="$7"
  local extra_loops="$8"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --dagger-batch-size 256 \
    --dagger-batches "${dagger_batches}" \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-rollout-cycles 4 \
    --evaluation-batch-size 256 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --ranks 32 256 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_weighted_shared_dagger() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local dagger_batches="$5"
  local dagger_rounds="$6"
  local evaluation_batches="$7"
  local extra_loops="$8"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --dagger-batch-size 256 \
    --dagger-batches "${dagger_batches}" \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-rollout-cycles 4 \
    --evaluation-batch-size 128 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --ranks 32 256 \
    --answer-repeats 8 28 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_horizon8_shared_dagger() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local dagger_batch_size="$5"
  local dagger_rounds="$6"
  local evaluation_batches="$7"
  local extra_loops="$8"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --dagger-batch-size "${dagger_batch_size}" \
    --dagger-batches 1 \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-rollout-cycles 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --ranks 256 \
    --answer-repeats 28 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_horizon16_shared_dagger() {
  local label="$1"
  local seed_base="$2"
  local calibration_batches="$3"
  local heldout_batches="$4"
  local dagger_batch_size="$5"
  local dagger_rounds="$6"
  local evaluation_batches="$7"
  local extra_loops="$8"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 256 \
    --heldout-batches "${heldout_batches}" \
    --dagger-batch-size "${dagger_batch_size}" \
    --dagger-batches 1 \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-rollout-cycles 16 \
    --evaluation-batch-size 64 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --ranks 256 \
    --answer-repeats 28 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_shared_power_eval() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  local map_label="${4:-w28_r256_round4}"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_power_eval \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "${map_label}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 256 \
    --batches 4 \
    --min-age 3 \
    --max-age 8 \
    --seed "${seed}" \
    --device cuda
}

run_adjacent_age() {
  local label="$1"
  local calibration_seed="$2"
  local evaluation_seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_adjacent_age \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches 4 \
    --evaluation-batch-size 128 \
    --evaluation-batches 8 \
    --extra-loops 64 \
    --min-source-age 3 \
    --max-source-age 8 \
    --ranks 32 256 \
    --answer-repeat 28 \
    --ridge 0.01 \
    --calibration-seed "${calibration_seed}" \
    --evaluation-seed "${evaluation_seed}" \
    --device cuda
}

run_map_ablation() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_map_ablation \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --seed "${seed}" \
    --device cuda
}

run_long_component_patch() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_long_component_patch \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --probe-cycles 16 24 32 40 48 64 \
    --seed "${seed}" \
    --device cuda
}

run_role_action() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_role_action \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --probe-cycles 2 8 16 24 32 48 64 \
    --seed "${seed}" \
    --device cuda
}

run_horizon8_rank_sweep() {
  local label="$1"
  local seed_base="$2"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches 4 \
    --heldout-batch-size 256 \
    --heldout-batches 4 \
    --dagger-batch-size 128 \
    --dagger-batches 1 \
    --dagger-rounds 4 \
    --dagger-rollout-cycles 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches 4 \
    --extra-loops 64 \
    --ranks 8 16 32 64 128 256 \
    --answer-repeats 28 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_initialization_gate() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_initialization_gate \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --seed "${seed}" \
    --device cuda
}

run_learned_initializer() {
  local label="$1"
  local feedback_artifact="$2"
  local seed_base="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_learned_initializer \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${feedback_artifact}" \
    --feedback-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches 4 \
    --heldout-batch-size 256 \
    --heldout-batches 4 \
    --evaluation-batch-size 128 \
    --evaluation-batches 8 \
    --extra-loops 64 \
    --answer-repeat 28 \
    --ridge 0.01 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --evaluation-seed "$((seed_base + 2))" \
    --device cuda
}

run_joint_single_map() {
  local label="$1"
  local feedback_artifact="$2"
  local initializer_artifact="$3"
  local seed_base="$4"
  local initializer_repeat="${5:-1}"
  local calibration_batches="${6:-4}"
  local evaluation_batches="${7:-8}"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_joint_single_map \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${feedback_artifact}" \
    --feedback-label "w28_r256_round4" \
    --initializer-artifact "${initializer_artifact}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches "${calibration_batches}" \
    --dagger-batch-size 64 \
    --dagger-batches 1 \
    --dagger-rounds 4 \
    --dagger-rollout-cycles 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops 64 \
    --answer-repeat 28 \
    --initializer-repeat "${initializer_repeat}" \
    --ridge 0.01 \
    --calibration-seed "${seed_base}" \
    --dagger-seed "$((seed_base + 1))" \
    --evaluation-seed "$((seed_base + 2))" \
    --device cuda
}

run_observability_audit() {
  local label="$1"
  local seed="$2"
  local one_step_operator="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_observability_audit \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --one-step-operator "${one_step_operator}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 256 \
    --batches 16 \
    --seed "${seed}" \
    --device cuda
}

run_per_node_lifespan() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_per_node_lifespan \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 16 \
    --extra-loops 64 \
    --seed "${seed}" \
    --device cuda
}

run_per_node_components() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_per_node_component \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 16 \
    --extra-loops 32 \
    --probe-cycles 2 4 8 16 32 \
    --seed "${seed}" \
    --device cuda
}

run_orbit_control() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_orbit_control \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 32 \
    --extra-loops 64 \
    --seed "${seed}" \
    --device cuda
}

run_horizon8_answer_weight_sweep() {
  local label="$1"
  local seed_base="$2"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_shared_position_dagger \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 256 \
    --calibration-batches 4 \
    --heldout-batch-size 256 \
    --heldout-batches 4 \
    --dagger-batch-size 128 \
    --dagger-batches 1 \
    --dagger-rounds 4 \
    --dagger-rollout-cycles 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches 8 \
    --extra-loops 64 \
    --ranks 256 \
    --answer-repeats 14 28 42 56 \
    --memory-efficient-full-rank \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --dagger-seed "$((seed_base + 2))" \
    --evaluation-seed "$((seed_base + 3))" \
    --ridge 0.01 \
    --device cuda
}

run_map_strength() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_map_ablation \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --seed "${seed}" \
    --executor-head 2 \
    --strengths -1 0 0.25 0.5 0.75 1 1.25 1.5 \
    --strength-only \
    --device cuda
}

run_map_strength_fine() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_map_ablation \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --seed "${seed}" \
    --executor-head 2 \
    --strengths 0.85 0.9 0.95 1 1.05 1.1 1.15 \
    --strength-only \
    --device cuda
}

run_probe_dose() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_probe_dose \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --map-artifact "${map_artifact}" \
    --map-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --probe-cycles 2 8 16 24 32 48 64 \
    --strengths -1 0 0.25 0.5 0.75 1 1.25 1.5 2 \
    --seed "${seed}" \
    --executor-head 2 \
    --device cuda
}

run_periodic_booster() {
  local label="$1"
  local map_artifact="$2"
  local seed_base="$3"
  local switch_cycle="${4:-32}"
  local period="${5:-32}"
  local dagger_rounds="${6:-0}"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_periodic_booster \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${map_artifact}" \
    --feedback-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 128 \
    --calibration-batches 8 \
    --heldout-batch-size 128 \
    --heldout-batches 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches 8 \
    --extra-loops 128 \
    --switch-cycle "${switch_cycle}" \
    --period "${period}" \
    --answer-repeat 28 \
    --ridge 0.01 \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --evaluation-seed "$((seed_base + 2))" \
    --dagger-rounds "${dagger_rounds}" \
    --dagger-batch-size 128 \
    --dagger-batches 4 \
    --dagger-rollout-cycles 96 \
    --dagger-seed "$((seed_base + 3))" \
    --device cuda
}

run_exact_renewal_scan() {
  local label="$1"
  local map_artifact="$2"
  local seed="$3"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_exact_renewal_scan \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${map_artifact}" \
    --feedback-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 128 \
    --periods 16 20 22 24 25 26 28 30 32 \
    --seed "${seed}" \
    --executor-head 2 \
    --device cuda
}

run_reset_age_cross_eval() {
  local label="$1"
  local feedback_artifact="$2"
  local booster24_artifact="$3"
  local booster32_artifact="$4"
  local seed="$5"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_reset_age_cross_eval \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${feedback_artifact}" \
    --feedback-label "w28_r256_round4" \
    --booster24-artifact "${booster24_artifact}" \
    --booster32-artifact "${booster32_artifact}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches 8 \
    --extra-loops 64 \
    --probe-cycles 24 32 48 64 \
    --seed "${seed}" \
    --executor-head 2 \
    --device cuda
}

run_role_conditioned_booster() {
  local label="$1"
  local map_artifact="$2"
  local seed_base="$3"
  local calibration_batches="${4:-8}"
  local role_rank="${5:-256}"
  local extra_loops="${6:-128}"
  local evaluation_batches="${7:-8}"
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_role_conditioned_booster \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${map_artifact}" \
    --feedback-label "w28_r256_round4" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --calibration-batch-size 128 \
    --calibration-batches "${calibration_batches}" \
    --heldout-batch-size 128 \
    --heldout-batches 8 \
    --evaluation-batch-size 128 \
    --evaluation-batches "${evaluation_batches}" \
    --extra-loops "${extra_loops}" \
    --switch-cycle 24 \
    --period 24 \
    --ridge 0.01 \
    --role-rank "${role_rank}" \
    --calibration-seed "${seed_base}" \
    --heldout-seed "$((seed_base + 1))" \
    --evaluation-seed "$((seed_base + 2))" \
    --device cuda
}

run_role_hybrid_eval() {
  local label="$1"
  local map_artifact="$2"
  local role_artifact="$3"
  local seed="$4"
  local mode="${5:-selected}"
  local extra_loops="${6:-128}"
  local batches="${7:-8}"
  local subset_args=()
  if [[ "${mode}" == "all" ]]; then
    subset_args+=(--all-subsets)
  fi
  "${PYTHON_BIN}" -u -m \
    reasoning_loop.graph_path_telomere_role_hybrid_eval \
    --checkpoint "${CHECKPOINT}" \
    --phase-summary "${PHASE_SUMMARY}" \
    --feedback-artifact "${map_artifact}" \
    --feedback-label "w28_r256_round4" \
    --role-artifact "${role_artifact}" \
    --out-dir "${OUTPUT_ROOT}/${label}" \
    --batch-size 128 \
    --batches "${batches}" \
    --extra-loops "${extra_loops}" \
    --first-cycle 24 \
    --period 24 \
    --seed "${seed}" \
    --executor-head 2 \
    "${subset_args[@]}" \
    --device cuda
}

write_manifest "running"
trap 'write_manifest "failed"' ERR

case "${ACTION}" in
  smoke)
    run_one "smoke" 74001 2 1 1 1 4
    ;;
  formal)
    run_one "formal_primary" 74101 16 8 4 2 16
    run_one "formal_replica" 75101 16 8 4 2 16
    ;;
  dagger-smoke)
    run_dagger "dagger_smoke" 76001 2 1 1 2 1 8
    ;;
  dagger-formal)
    run_dagger "dagger_primary" 76101 16 8 4 4 4 32
    run_dagger "dagger_replica" 77101 16 8 4 4 4 32
    ;;
  position-smoke)
    run_position_circuit "position_smoke" 78001 1 8
    ;;
  position-formal)
    run_position_circuit "position_primary" 78101 4 32
    run_position_circuit "position_replica" 79101 4 32
    ;;
  complement-formal)
    run_position_circuit "complement_primary" 88101 4 32
    run_position_circuit "complement_replica" 89101 4 32
    ;;
  shared-smoke)
    run_shared_dagger "shared_smoke" 80001 2 1 1 2 1 8
    ;;
  shared-formal)
    run_shared_dagger "shared_primary" 80101 8 4 2 4 4 32
    run_shared_dagger "shared_replica" 81101 8 4 2 4 4 32
    ;;
  weighted-smoke)
    run_weighted_shared_dagger "weighted_smoke" 82001 1 1 1 2 2 8
    ;;
  weighted-formal)
    run_weighted_shared_dagger "weighted_primary" 82101 4 4 1 4 8 32
    run_weighted_shared_dagger "weighted_replica" 83101 4 4 1 4 8 32
    ;;
  horizon8-smoke)
    run_horizon8_shared_dagger "horizon8_smoke" 84001 2 1 64 2 2 32
    ;;
  horizon8-formal)
    run_horizon8_shared_dagger "horizon8_primary" 84101 4 4 128 4 8 64
    run_horizon8_shared_dagger "horizon8_replica" 85101 4 4 128 4 8 64
    ;;
  horizon16-smoke)
    run_horizon16_shared_dagger "horizon16_smoke" 86001 2 1 32 2 4 64
    ;;
  horizon16-formal)
    run_horizon16_shared_dagger "horizon16_primary" 86101 4 4 64 4 16 128
    run_horizon16_shared_dagger "horizon16_replica" 87101 4 4 64 4 16 128
    ;;
  power-formal)
    run_shared_power_eval \
      "power_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      90101
    run_shared_power_eval \
      "power_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      91101
    ;;
  adjacent-formal)
    run_adjacent_age "adjacent_primary" 92101 92102
    run_shared_power_eval \
      "adjacent_power_primary" \
      "${OUTPUT_ROOT}/adjacent_primary/adjacent_age_maps.pt" \
      92103 \
      "adjacent_w28_r256"
    run_adjacent_age "adjacent_replica" 93101 93102
    run_shared_power_eval \
      "adjacent_power_replica" \
      "${OUTPUT_ROOT}/adjacent_replica/adjacent_age_maps.pt" \
      93103 \
      "adjacent_w28_r256"
    ;;
  ablation-formal)
    run_map_ablation \
      "ablation_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      94101
    run_map_ablation \
      "ablation_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      95101
    ;;
  component-long-formal)
    run_long_component_patch \
      "component_long_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      96101
    run_long_component_patch \
      "component_long_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      97101
    ;;
  role-formal)
    run_role_action \
      "role_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      98101
    run_role_action \
      "role_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      99101
    ;;
  rank-formal)
    run_horizon8_rank_sweep "rank_primary" 100101
    run_horizon8_rank_sweep "rank_replica" 101101
    ;;
  initialization-formal)
    run_initialization_gate \
      "initialization_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      102101
    run_initialization_gate \
      "initialization_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      103101
    ;;
  learned-initializer-formal)
    run_learned_initializer \
      "learned_initializer_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      104101
    run_learned_initializer \
      "learned_initializer_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      105101
    ;;
  joint-single-formal)
    run_joint_single_map \
      "joint_single_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_primary/learned_initializer.pt" \
      106101
    run_joint_single_map \
      "joint_single_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_replica/learned_initializer.pt" \
      107101
    ;;
  joint-weight-formal)
    run_joint_single_map \
      "joint_weight1_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_primary/learned_initializer.pt" \
      108101 1 2 4
    run_joint_single_map \
      "joint_weight1_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_replica/learned_initializer.pt" \
      109101 1 2 4
    run_joint_single_map \
      "joint_weight2_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_primary/learned_initializer.pt" \
      110101 2 2 4
    run_joint_single_map \
      "joint_weight2_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_replica/learned_initializer.pt" \
      111101 2 2 4
    run_joint_single_map \
      "joint_weight4_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_primary/learned_initializer.pt" \
      112101 4 2 4
    run_joint_single_map \
      "joint_weight4_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/learned_initializer_replica/learned_initializer.pt" \
      113101 4 2 4
    ;;
  observability-formal)
    run_observability_audit \
      "observability_primary" \
      114101 \
      "/data/paperexperiment/graph_path_telomere_simple_one_step_D8L8_20260730/formal_primary/simple_one_step_R.pt"
    run_observability_audit \
      "observability_replica" \
      115101 \
      "/data/paperexperiment/graph_path_telomere_simple_one_step_D8L8_20260730/formal_replica/simple_one_step_R.pt"
    ;;
  per-node-lifespan-formal)
    run_per_node_lifespan \
      "per_node_lifespan_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      116101
    run_per_node_lifespan \
      "per_node_lifespan_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      117101
    ;;
  per-node-component-formal)
    run_per_node_components \
      "per_node_component_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      118101
    run_per_node_components \
      "per_node_component_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      119101
    ;;
  orbit-control-formal)
    run_orbit_control \
      "orbit_control_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      120101
    run_orbit_control \
      "orbit_control_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      121101
    ;;
  orbit-control-h16-formal)
    run_orbit_control \
      "orbit_control_h16_primary" \
      "${OUTPUT_ROOT}/horizon16_primary/shared_position_maps.pt" \
      122101
    run_orbit_control \
      "orbit_control_h16_replica" \
      "${OUTPUT_ROOT}/horizon16_replica/shared_position_maps.pt" \
      123101
    ;;
  answer-weight-h8-formal)
    run_horizon8_answer_weight_sweep \
      "answer_weight_h8_primary" \
      124101
    run_horizon8_answer_weight_sweep \
      "answer_weight_h8_replica" \
      125101
    ;;
  map-strength-formal)
    run_map_strength \
      "map_strength_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      126101
    run_map_strength \
      "map_strength_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      127101
    ;;
  map-strength-fine-formal)
    run_map_strength_fine \
      "map_strength_fine_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      128101
    run_map_strength_fine \
      "map_strength_fine_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      129101
    ;;
  probe-dose-formal)
    run_probe_dose \
      "probe_dose_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      130101
    run_probe_dose \
      "probe_dose_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      131101
    ;;
  periodic-booster-formal)
    run_periodic_booster \
      "periodic_booster_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      132101
    run_periodic_booster \
      "periodic_booster_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      133101
    ;;
  periodic-booster24-formal)
    run_periodic_booster \
      "periodic_booster24_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      134101 \
      24 \
      24
    run_periodic_booster \
      "periodic_booster24_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      135101 \
      24 \
      24
    ;;
  periodic-booster24-dagger-formal)
    run_periodic_booster \
      "periodic_booster24_dagger_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      136101 \
      24 \
      24 \
      4
    run_periodic_booster \
      "periodic_booster24_dagger_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      137101 \
      24 \
      24 \
      4
    ;;
  exact-renewal-scan-formal)
    run_exact_renewal_scan \
      "exact_renewal_scan_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      138101
    run_exact_renewal_scan \
      "exact_renewal_scan_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      139101
    ;;
  reset-age-cross-formal)
    run_reset_age_cross_eval \
      "reset_age_cross_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/periodic_booster24_primary/periodic_booster.pt" \
      "${OUTPUT_ROOT}/periodic_booster_primary/periodic_booster.pt" \
      154101
    run_reset_age_cross_eval \
      "reset_age_cross_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/periodic_booster24_replica/periodic_booster.pt" \
      "${OUTPUT_ROOT}/periodic_booster_replica/periodic_booster.pt" \
      155101
    ;;
  role-booster24-formal)
    run_role_conditioned_booster \
      "role_booster24_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      140101
    run_role_conditioned_booster \
      "role_booster24_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      141101
    ;;
  role-booster24-large-formal)
    run_role_conditioned_booster \
      "role_booster24_large_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      142101 \
      32
    run_role_conditioned_booster \
      "role_booster24_large_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      143101 \
      32
    ;;
  role-booster24-rank25-formal)
    run_role_conditioned_booster \
      "role_booster24_rank25_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      144101 \
      32 \
      25
    run_role_conditioned_booster \
      "role_booster24_rank25_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      145101 \
      32 \
      25
    ;;
  role-hybrid24-formal)
    run_role_hybrid_eval \
      "role_hybrid24_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_rank25_primary/role_conditioned_booster.pt" \
      146101
    run_role_hybrid_eval \
      "role_hybrid24_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_rank25_replica/role_conditioned_booster.pt" \
      147101
    ;;
  role-factorial24-formal)
    run_role_hybrid_eval \
      "role_factorial24_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_rank25_primary/role_conditioned_booster.pt" \
      148101 \
      all \
      64 \
      8
    run_role_hybrid_eval \
      "role_factorial24_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_rank25_replica/role_conditioned_booster.pt" \
      149101 \
      all \
      64 \
      8
    ;;
  role-factorial24-full-formal)
    run_role_hybrid_eval \
      "role_factorial24_full_primary" \
      "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_large_primary/role_conditioned_booster.pt" \
      150101 \
      all \
      64 \
      8
    run_role_hybrid_eval \
      "role_factorial24_full_replica" \
      "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
      "${OUTPUT_ROOT}/role_booster24_large_replica/role_conditioned_booster.pt" \
      151101 \
      all \
      64 \
      8
    ;;
  role-rank-sweep24-formal)
    for role_rank in 0 8 16 25 32 64 128 256; do
      run_role_conditioned_booster \
        "role_rank_sweep24_r${role_rank}_primary" \
        "${OUTPUT_ROOT}/horizon8_primary/shared_position_maps.pt" \
        152101 \
        16 \
        "${role_rank}" \
        64 \
        8
      run_role_conditioned_booster \
        "role_rank_sweep24_r${role_rank}_replica" \
        "${OUTPUT_ROOT}/horizon8_replica/shared_position_maps.pt" \
        153101 \
        16 \
        "${role_rank}" \
        64 \
        8
    done
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac

trap - ERR
write_manifest "complete"
