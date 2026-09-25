#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu="$1"
out_root="/data/wujiaju/graph_path_induction_contrast_20260726/raw"
log_root="/data/wujiaju/logs/graph_path_induction_contrast_20260726/analysis"
manifest="/data/wujiaju/graph_path_induction_contrast_20260726/analysis_manifest.txt"
mkdir -p "${out_root}" "${log_root}"
printf 'status=running\npid=%s\nphysical_gpu=%s\nstarted=%s\noutput=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" "${out_root}" >"${manifest}"

checkpoint_for() {
  local family="$1"
  local seed="$2"
  local depth loops
  depth="${family#D}"
  depth="${depth%%_*}"
  loops="${family##*_L}"
  if [[ "${family}" == "D8_L8" ]]; then
    echo "/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/best.pt"
  elif [[ "${seed}" == "0" ]]; then
    echo "/data/wujiaju/LooPlus/results/graph_path_reg_wd03_long_20260706/graphpath_N8_D${depth}_d256_B2_L${loops}_seed0/best.pt"
  elif [[ "${seed}" -le 2 ]]; then
    echo "/data/wujiaju/graph_path_depth_circuit_20260725/training/${family}_seed${seed}/graphpath_N8_D${depth}_d256_B2_L${loops}_seed${seed}/best.pt"
  else
    echo "/data/wujiaju/graph_path_functional_multiseed_20260725/training/${family}_seed${seed}/graphpath_N8_D${depth}_d256_B2_L${loops}_seed${seed}/best.pt"
  fi
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/wujiaju/LooPlus

for family in D6_L6 D6_L8 D8_L6 D8_L8; do
  run_args=()
  for seed in 0 1 2 3 4 5 6 7; do
    checkpoint="$(checkpoint_for "${family}" "${seed}")"
    if [[ ! -s "${checkpoint}" ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    run_args+=(--run "${family}_seed${seed}=${checkpoint}")
  done
  {
    echo "START $(date --iso-8601=seconds)"
    echo "PHYSICAL_GPU ${gpu}"
    /data/wujiaju/.venvs/loopreasoner/bin/python -u \
      -m reasoning_loop.graph_path_induction_contrast \
      "${run_args[@]}" \
      --out-dir "${out_root}/${family}" \
      --batch-size 128 \
      --max-bundle-edges 4 \
      --device cuda \
      --force
    echo "COMPLETE $(date --iso-8601=seconds)"
  } >"${log_root}/${family}.log" 2>&1
done
printf 'status=complete\npid=%s\nphysical_gpu=%s\ncompleted=%s\noutput=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" "${out_root}" >"${manifest}"
