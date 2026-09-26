#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:?usage: $0 PHYSICAL_GPU}"
project_root="${PROJECT_ROOT:-/data/paperexperiment/LooPlus}"
output_root="${OUTPUT_ROOT:-/data/paperexperiment/graph_path_rejuvenation_multimodel_20260730/formal}"
python_bin="${PYTHON_BIN:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
experiment_seed="${EXPERIMENT_SEED:-2026073001}"
lifespan_root="/data/paperexperiment/graph_path_telomere_lifespan_extension_20260729/formal"
compression_root="/data/paperexperiment/graph_path_compression_circuit_20260725/training"
functional_root="/data/paperexperiment/graph_path_functional_multiseed_20260725/training"

mkdir -p "${output_root}"
cd "${project_root}"

CUDA_VISIBLE_DEVICES="${physical_gpu}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${project_root}" \
"${python_bin}" -u -m reasoning_loop.graph_path_rejuvenation_multimodel \
  --run "D8_L6_seed6=${functional_root}/D8_L6_seed6/graphpath_N8_D8_d256_B2_L6_seed6/best.pt,${lifespan_root}/D8_L6_seed6/summary.json" \
  --run "D8_L8_seed0=${compression_root}/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt,${lifespan_root}/D8_L8_seed0/summary.json" \
  --run "D8_L8_seed1=${compression_root}/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt,${lifespan_root}/D8_L8_seed1/summary.json" \
  --run "D8_L8_seed2=${compression_root}/D8_L8_seed2/graphpath_N8_D8_d256_B2_L8_seed2/best.pt,${lifespan_root}/D8_L8_seed2/summary.json" \
  --run "D8_L8_seed5=${compression_root}/D8_L8_seed5/graphpath_N8_D8_d256_B2_L8_seed5/best.pt,${lifespan_root}/D8_L8_seed5/summary.json" \
  --out-dir "${output_root}" \
  --calibration-batch-size 256 \
  --calibration-batches 8 \
  --validation-size 512 \
  --discovery-size 256 \
  --evaluation-size 512 \
  --lifespan-batch-size 128 \
  --lifespan-batches 4 \
  --extra-loops 8 \
  --ridge-values 0.00001 0.0001 0.001 0.01 0.1 1.0 \
  --ranks 0 1 2 4 8 16 32 64 128 \
  --max-age-rank 128 \
  --random-circuit-subsets 16 \
  --cuda-memory-fraction 0.045 \
  --seed "${experiment_seed}" \
  --device cuda
