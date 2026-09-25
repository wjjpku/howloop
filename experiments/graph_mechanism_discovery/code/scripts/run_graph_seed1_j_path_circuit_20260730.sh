#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:?usage: $0 PHYSICAL_GPU}"
project_root="${PROJECT_ROOT:-/data/wujiaju/LooPlus}"
output_root="${OUTPUT_ROOT:-/data/wujiaju/graph_path_seed1_j_path_circuit_20260730/formal}"
python_bin="${PYTHON_BIN:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
checkpoint="/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt"
matrix="/data/wujiaju/graph_path_rejuvenation_multimodel_20260730/formal_causal/D8_L8_seed1/single_rejuvenation_matrix.pt"

mkdir -p "${output_root}"
cd "${project_root}"

CUDA_VISIBLE_DEVICES="${physical_gpu}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${project_root}" \
"${python_bin}" -u -m reasoning_loop.graph_path_seed1_j_path_circuit \
  --checkpoint "${checkpoint}" \
  --matrix "${matrix}" \
  --out-dir "${output_root}" \
  --batch-size 512 \
  --seed 2026074101 \
  --seed 2026075101 \
  --cuda-memory-fraction 0.045 \
  --device cuda
