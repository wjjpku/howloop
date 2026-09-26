#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:?usage: $0 PHYSICAL_GPU}"
project_root="${PROJECT_ROOT:-/data/paperexperiment/LooPlus}"
output_root="${OUTPUT_ROOT:-/data/paperexperiment/graph_path_seed1_lowrank_telomere_20260730/formal}"
python_bin="${PYTHON_BIN:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
checkpoint="/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt"
lifespan_summary="/data/paperexperiment/graph_path_telomere_lifespan_extension_20260729/formal/D8_L8_seed1/summary.json"

mkdir -p "${output_root}"
cd "${project_root}"

CUDA_VISIBLE_DEVICES="${physical_gpu}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${project_root}" \
"${python_bin}" -u -m reasoning_loop.graph_path_seed1_lowrank_telomere \
  --checkpoint "${checkpoint}" \
  --lifespan-summary "${lifespan_summary}" \
  --out-dir "${output_root}" \
  --ranks 0 1 2 4 8 16 32 64 128 256 \
  --factor-seeds 0 1 \
  --calibration-batch-size 256 \
  --calibration-batches 8 \
  --validation-size 512 \
  --evaluation-size 512 \
  --steps 600 \
  --batch-size 512 \
  --learning-rate 0.01 \
  --weight-decay 0.00001 \
  --validation-interval 50 \
  --cuda-memory-fraction 0.045 \
  --seed 2026076101 \
  --device cuda
