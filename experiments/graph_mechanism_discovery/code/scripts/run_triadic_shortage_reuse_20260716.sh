#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PHASE="${PHASE:-smoke}"
GPU_LIST_STRING="${GPU_LIST:-0}"
GPU_LIST_STRING="${GPU_LIST_STRING//,/ }"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/triadic_shortage_reuse_20260716}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
FORMAL_STEPS="${FORMAL_STEPS:-10000}"
SMOKE_STEPS="${SMOKE_STEPS:-200}"

read -r -a GPUS <<< "${GPU_LIST_STRING}"
if [[ ${#GPUS[@]} -lt 1 || ${#GPUS[@]} -gt 3 ]]; then
  echo "GPU_LIST must contain between one and three GPU ids" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/runs" "${LOG_DIR}" "${OUTPUT_DIR}/analysis" "${OUTPUT_DIR}/manifests"
cd "${PROJECT_ROOT}"
shasum -a 256 \
  reasoning_loop/triadic_shortage*.py \
  tests/test_triadic_shortage*.py \
  scripts/run_triadic_shortage_reuse_20260716.sh \
  > "${OUTPUT_DIR}/manifests/source_${PHASE}.sha256"

if [[ "${PHASE}" == "analyze" ]]; then
  while IFS= read -r checkpoint; do
    run_dir="$(dirname "${checkpoint}")"
    run_name="$(basename "${run_dir}")"
    condition="$(${PYTHON_BIN} -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["condition"])' "${checkpoint}")"
    if [[ "${condition}" != "sequential" && "${condition}" != "sequential_shuffled" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
      -m reasoning_loop.triadic_shortage_diagnostics \
      --checkpoint "${checkpoint}" \
      --out-dir "${OUTPUT_DIR}/analysis/diagnostics/${run_name}" \
      --device cuda \
      > "${LOG_DIR}/analyze_${run_name}.log" 2>&1
  done < <(find "${OUTPUT_DIR}/runs" -name final.pt -type f | sort)
  PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
    -m reasoning_loop.triadic_shortage_aggregate \
    --run-root "${OUTPUT_DIR}" \
    --out-dir "${OUTPUT_DIR}/analysis/aggregate"
  exit 0
fi

if [[ "${PHASE}" == "circuit" ]]; then
  for seed in 0 1 2; do
    run_name="info_sequential_d64_L6_seed${seed}"
    checkpoint="${OUTPUT_DIR}/runs/${run_name}/final.pt"
    [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
      -m reasoning_loop.triadic_shortage_circuit \
      --checkpoint "${checkpoint}" \
      --out-dir "${OUTPUT_DIR}/analysis/circuits/${run_name}" \
      --device cuda \
      > "${LOG_DIR}/circuit_${run_name}.log" 2>&1
  done
  PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
    -m reasoning_loop.triadic_shortage_aggregate \
    --run-root "${OUTPUT_DIR}" \
    --out-dir "${OUTPUT_DIR}/analysis/aggregate"
  exit 0
fi

if [[ "${PHASE}" == "conversion_analyze" ]]; then
  mapfile -t conversion_jobs < <(
    OUTPUT_DIR="${OUTPUT_DIR}" PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -c '
from pathlib import Path
from reasoning_loop.triadic_shortage_conversion import discover_conversion_checkpoints
import os
for name, checkpoint in discover_conversion_checkpoints(Path(os.environ["OUTPUT_DIR"])):
    print(f"{name}|{checkpoint}")
'
  )
  run_conversion_worker() {
    local worker_index="$1"
    local gpu="$2"
    local worker_count="$3"
    local job_index name checkpoint
    for ((job_index=worker_index; job_index<${#conversion_jobs[@]}; job_index+=worker_count)); do
      IFS='|' read -r name checkpoint <<< "${conversion_jobs[job_index]}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
        -m reasoning_loop.triadic_shortage_conversion \
        --checkpoint "${checkpoint}" \
        --out-dir "${OUTPUT_DIR}/analysis/conversion/${name}" \
        --device cuda \
        > "${LOG_DIR}/conversion_${name}.log" 2>&1
    done
  }
  conversion_pids=()
  for ((worker_index=0; worker_index<${#GPUS[@]}; worker_index+=1)); do
    run_conversion_worker "${worker_index}" "${GPUS[worker_index]}" "${#GPUS[@]}" &
    conversion_pids+=("$!")
  done
  for pid in "${conversion_pids[@]}"; do wait "${pid}"; done
  PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
    -m reasoning_loop.triadic_shortage_conversion \
    --aggregate-root "${OUTPUT_DIR}/analysis/conversion" \
    --out-dir "${OUTPUT_DIR}/analysis/conversion/aggregate"
  exit 0
fi

JOBS=()

add_job() {
  local condition="$1"
  local architecture="$2"
  local loops="$3"
  local width="$4"
  local seed="$5"
  local total_steps="$6"
  local stop_step="$7"
  local run_name="$8"
  local resume_path="${9:-none}"
  JOBS+=("${condition}|${architecture}|${loops}|${width}|${seed}|${total_steps}|${stop_step}|${run_name}|${resume_path}")
}

case "${PHASE}" in
  smoke)
    for condition in full sequential sequential_shuffled; do
      add_job "${condition}" looped 6 64 0 "${SMOKE_STEPS}" "${SMOKE_STEPS}" "smoke_${condition}_seed0"
    done
    ;;
  information)
    for condition in full full_once sequential sequential_shuffled; do
      for seed in 0 1 2; do
        add_job "${condition}" looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "info_${condition}_d64_L6_seed${seed}"
      done
    done
    ;;
  compute)
    for width in 16 24 32 48 64; do
      for seed in 0 1 2; do
        add_job full looped 1 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_L1_d${width}_seed${seed}"
        add_job full looped 6 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_L1x6_d${width}_seed${seed}"
        add_job full unshared 6 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_S6_d${width}_seed${seed}"
      done
    done
    ;;
  compute_narrow)
    for width in 4 8 12; do
      for seed in 0 1 2; do
        add_job full looped 1 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_L1_d${width}_seed${seed}"
        add_job full looped 6 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_L1x6_d${width}_seed${seed}"
        add_job full unshared 6 "${width}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "compute_S6_d${width}_seed${seed}"
      done
    done
    ;;
  confirm)
    if [[ -z "${SELECTED_WIDTH:-}" ]]; then
      echo "SELECTED_WIDTH is required for confirm" >&2
      exit 2
    fi
    for seed in 3 4 5 6 7; do
      add_job full looped 1 "${SELECTED_WIDTH}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "confirm_L1_d${SELECTED_WIDTH}_seed${seed}"
      add_job full looped 6 "${SELECTED_WIDTH}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "confirm_L1x6_d${SELECTED_WIDTH}_seed${seed}"
      add_job full unshared 6 "${SELECTED_WIDTH}" "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "confirm_S6_d${SELECTED_WIDTH}_seed${seed}"
      add_job sequential looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "confirm_sequential_d64_L6_seed${seed}"
      add_job sequential_shuffled looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "confirm_sequential_shuffled_d64_L6_seed${seed}"
    done
    ;;
  switch)
    for seed in 0 1 2; do
      full_checkpoint="${OUTPUT_DIR}/runs/info_full_d64_L6_seed${seed}/checkpoint_step_0004000.pt"
      sequential_checkpoint="${OUTPUT_DIR}/runs/info_sequential_shuffled_d64_L6_seed${seed}/checkpoint_step_0004000.pt"
      [[ -f "${full_checkpoint}" ]] || { echo "missing ${full_checkpoint}" >&2; exit 2; }
      [[ -f "${sequential_checkpoint}" ]] || { echo "missing ${sequential_checkpoint}" >&2; exit 2; }
      add_job sequential_shuffled looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "switch_full_to_sequential_seed${seed}" "${full_checkpoint}"
      add_job full looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "switch_sequential_to_full_seed${seed}" "${sequential_checkpoint}"
      add_job full looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "switch_full_to_full_seed${seed}" "${full_checkpoint}"
      add_job sequential_shuffled looped 6 64 "${seed}" "${FORMAL_STEPS}" "${FORMAL_STEPS}" "switch_sequential_to_sequential_seed${seed}" "${sequential_checkpoint}"
    done
    ;;
  *)
    echo "unknown PHASE=${PHASE}" >&2
    exit 2
    ;;
esac

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local worker_count="$3"
  local job_index
  for ((job_index=worker_index; job_index<${#JOBS[@]}; job_index+=worker_count)); do
    IFS='|' read -r condition architecture loops width seed total_steps stop_step run_name resume_path <<< "${JOBS[job_index]}"
    command=(
      "${PYTHON_BIN}" -m reasoning_loop.triadic_shortage_train
      --p 17
      --condition "${condition}"
      --architecture "${architecture}"
      --d-model "${width}"
      --n-heads 4
      --d-mlp "$((2 * width))"
      --loops "${loops}"
      --seed "${seed}"
      --steps "${total_steps}"
      --stop-step "${stop_step}"
      --device cuda
      --run-name "${run_name}"
      --out-dir "${OUTPUT_DIR}/runs"
    )
    if [[ "${resume_path}" != "none" ]]; then
      command+=(--resume "${resume_path}")
    fi
    echo "[$(date -Is)] gpu=${gpu} run=${run_name}" | tee -a "${LOG_DIR}/launch_${PHASE}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}" "${command[@]}" \
      > "${LOG_DIR}/${run_name}.log" 2>&1
  done
}

worker_count="${#GPUS[@]}"
pids=()
for ((worker_index=0; worker_index<worker_count; worker_index+=1)); do
  run_worker "${worker_index}" "${GPUS[worker_index]}" "${worker_count}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
  -m reasoning_loop.triadic_shortage_aggregate \
  --run-root "${OUTPUT_DIR}" \
  --out-dir "${OUTPUT_DIR}/analysis/aggregate"
