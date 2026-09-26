#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift
first_seed="$1"
root="/data/paperexperiment/graph_path_induction_contrast_20260726"
log_root="/data/paperexperiment/logs/graph_path_induction_contrast_20260726/paired_stretch_analysis"
manifest="${root}/paired_stretch_analysis_manifest_seed${first_seed}.txt"
mkdir -p "${log_root}"
printf 'status=running\npid=%s\nphysical_gpu=%s\nstarted=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"

run_args=()
for seed in "$@"; do
  for loops in 6 8; do
    checkpoint="${root}/training_stretch/D6_L${loops}_pairseed${seed}/graphpath_N8_D6_d256_B2_L${loops}_seed${seed}/final.pt"
    if [[ ! -s "${checkpoint}" ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    run_args+=(--run "D6_L${loops}_seed${seed}=${checkpoint}")
  done
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/paperexperiment/LooPlus

{
  echo "START induction $(date --iso-8601=seconds)"
  /data/paperexperiment/.venvs/loopreasoner/bin/python -u \
    -m reasoning_loop.graph_path_induction_contrast \
    "${run_args[@]}" \
    --out-dir "${root}/paired_stretch_induction_raw" \
    --batch-size 256 \
    --max-bundle-edges 4 \
    --device cuda \
    --force
  echo "COMPLETE induction $(date --iso-8601=seconds)"
} >"${log_root}/induction.log" 2>&1

{
  echo "START functional $(date --iso-8601=seconds)"
  /data/paperexperiment/.venvs/loopreasoner/bin/python -u \
    -m reasoning_loop.graph_path_functional_circuit \
    "${run_args[@]}" \
    --out-dir "${root}/paired_stretch_functional_raw" \
    --batch-size 256 \
    --top-k 64 \
    --device cuda \
    --force
  echo "COMPLETE functional $(date --iso-8601=seconds)"
} >"${log_root}/functional.log" 2>&1

{
  echo "START compression $(date --iso-8601=seconds)"
  /data/paperexperiment/.venvs/loopreasoner/bin/python -u \
    -m reasoning_loop.graph_path_compression_circuit \
    "${run_args[@]}" \
    --out-dir "${root}/paired_stretch_compression_raw" \
    --batch-size 256 \
    --batches 8 \
    --extra-loops 4 \
    --device cuda
  echo "COMPLETE compression $(date --iso-8601=seconds)"
} >"${log_root}/compression.log" 2>&1

printf 'status=complete\npid=%s\nphysical_gpu=%s\ncompleted=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"
