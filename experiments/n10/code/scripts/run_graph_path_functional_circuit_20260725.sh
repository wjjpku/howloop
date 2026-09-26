#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?usage: $0 FAMILY PHYSICAL_GPU}"
PHYSICAL_GPU="${2:?usage: $0 FAMILY PHYSICAL_GPU}"
REPO_DIR="/data/paperexperiment/LooPlus"
PYTHON_BIN="/data/paperexperiment/.venvs/loopreasoner/bin/python"
OUTPUT_ROOT="/data/paperexperiment/graph_path_functional_circuit_20260725/raw"

case "${FAMILY}" in
  D6_L6)
    RUNS=(
      "D6_L6_seed0=/data/paperexperiment/LooPlus/results/graph_path_reg_wd03_long_20260706/graphpath_N8_D6_d256_B2_L6_seed0/best.pt"
      "D6_L6_seed1=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D6_L6_seed1/graphpath_N8_D6_d256_B2_L6_seed1/best.pt"
      "D6_L6_seed2=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D6_L6_seed2/graphpath_N8_D6_d256_B2_L6_seed2/best.pt"
    )
    ;;
  D6_L8)
    RUNS=(
      "D6_L8_seed0=/data/paperexperiment/LooPlus/results/graph_path_reg_wd03_long_20260706/graphpath_N8_D6_d256_B2_L8_seed0/best.pt"
      "D6_L8_seed1=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D6_L8_seed1/graphpath_N8_D6_d256_B2_L8_seed1/best.pt"
      "D6_L8_seed2=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D6_L8_seed2/graphpath_N8_D6_d256_B2_L8_seed2/best.pt"
    )
    ;;
  D8_L6)
    RUNS=(
      "D8_L6_seed0=/data/paperexperiment/LooPlus/results/graph_path_reg_wd03_long_20260706/graphpath_N8_D8_d256_B2_L6_seed0/best.pt"
      "D8_L6_seed1=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D8_L6_seed1/graphpath_N8_D8_d256_B2_L6_seed1/best.pt"
      "D8_L6_seed2=/data/paperexperiment/graph_path_depth_circuit_20260725/training/D8_L6_seed2/graphpath_N8_D8_d256_B2_L6_seed2/best.pt"
    )
    ;;
  *)
    echo "unknown family: ${FAMILY}" >&2
    exit 2
    ;;
esac

RUN_ARGS=()
for run in "${RUNS[@]}"; do
  RUN_ARGS+=(--run "${run}")
done

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/${FAMILY}"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${PYTHON_BIN}" \
  -m reasoning_loop.graph_path_functional_circuit \
  "${RUN_ARGS[@]}" \
  --out-dir "${OUTPUT_ROOT}/${FAMILY}" \
  --batch-size 256 \
  --top-k 64 \
  --seed 20260725 \
  --device cuda \
  --force
