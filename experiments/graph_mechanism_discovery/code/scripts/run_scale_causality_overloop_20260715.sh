#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/wujiaju/LooPlus}"
PYTHON="${PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wujiaju/scale_causality_overloop_20260715}"
LOG_DIR="${LOG_DIR:-/data/wujiaju/logs/scale_causality_overloop_20260715}"
NONE_DIR="${NONE_DIR:-/data/wujiaju/post_convergence_rmsnorm_raw_20260714}"
G1_DIR="${G1_DIR:-/data/wujiaju/post_convergence_innerg1_outerg1_20260714}"
GPU_LIST_STRING="${GPU_LIST:-1 2 3}"
FORCE="${FORCE:-0}"
read -r -a GPUS <<< "${GPU_LIST_STRING}"

if [[ ${#GPUS[@]} -lt 1 || ${#GPUS[@]} -gt 3 ]]; then
  echo "GPU_LIST must contain between one and three GPU ids" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${REPO}"
export MPLBACKEND=Agg

wait_batch() {
  local pid
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
  PIDS=()
}

run_phase() {
  local condition="$1"
  local experiment_dir="$2"
  local growth_schedule="$3"
  shift 3
  local seeds=("$@")
  local deltas=(0 10000 16000 20000)
  local seed delta gpu log_path
  local job_index=0
  PIDS=()
  for seed in "${seeds[@]}"; do
    for delta in "${deltas[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      log_path="${LOG_DIR}/${condition}_seed${seed}_delta${delta}.log"
      args=(
        -m small_modadd.scale_causality_overloop
        --condition "${condition}"
        --experiment-dir "${experiment_dir}"
        --output-dir "${OUTPUT_DIR}"
        --seeds "${seed}"
        --deltas "${delta}"
        --max-loops 100
        --mode all
        --device cuda
        --no-merge
      )
      if [[ -n "${growth_schedule}" ]]; then
        args+=(--growth-schedule "${growth_schedule}")
      fi
      if [[ "${FORCE}" == "1" ]]; then
        args+=(--force)
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${args[@]}" >"${log_path}" 2>&1 &
      PIDS+=("$!")
      job_index=$((job_index + 1))
      if [[ ${#PIDS[@]} -eq ${#GPUS[@]} ]]; then
        wait_batch
      fi
    done
  done
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    wait_batch
  fi
}

merge_results() {
  "${PYTHON}" -m small_modadd.scale_causality_overloop \
    --condition none \
    --experiment-dir "${NONE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --merge-only
}

phase="${1:-all}"
case "${phase}" in
  none)
    run_phase none "${NONE_DIR}" "" 0 1 2 3 4 5
    merge_results
    ;;
  g1)
    run_phase g1 "${G1_DIR}" "${OUTPUT_DIR}/none_growth_schedule.json" 0 1 3 5
    merge_results
    ;;
  merge)
    merge_results
    ;;
  all)
    run_phase none "${NONE_DIR}" "" 0 1 2 3 4 5
    merge_results
    run_phase g1 "${G1_DIR}" "${OUTPUT_DIR}/none_growth_schedule.json" 0 1 3 5
    merge_results
    ;;
  *)
    echo "usage: $0 [none|g1|merge|all]" >&2
    exit 2
    ;;
esac
