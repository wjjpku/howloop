#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FAMILY PHYSICAL_GPU" >&2
  exit 2
fi

family="$1"
physical_gpu="$2"
repo_dir="/data/paperexperiment/LooPlus"
training_root="/data/paperexperiment/graph_path_functional_multiseed_20260725/training"
output_root="/data/paperexperiment/graph_path_functional_multiseed_20260725/analysis/raw"

case "${family}" in
  D6_L6)
    max_depth=6
    loops=6
    ;;
  D6_L8)
    max_depth=6
    loops=8
    ;;
  D8_L6)
    max_depth=8
    loops=6
    ;;
  *)
    echo "unknown family: ${family}" >&2
    exit 2
    ;;
esac

run_args=()
for seed in 3 4 5 6 7; do
  checkpoint="${training_root}/${family}_seed${seed}/graphpath_N8_D${max_depth}_d256_B2_L${loops}_seed${seed}/best.pt"
  test -s "${checkpoint}"
  run_args+=(--run "${family}_seed${seed}=${checkpoint}")
done

mkdir -p "${output_root}/${family}"
cd "${repo_dir}"
CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  /data/paperexperiment/.venvs/loopreasoner/bin/python \
  -m reasoning_loop.graph_path_functional_circuit \
  "${run_args[@]}" \
  --out-dir "${output_root}/${family}" \
  --batch-size 256 \
  --top-k 64 \
  --seed 20260725 \
  --device cuda \
  --force
