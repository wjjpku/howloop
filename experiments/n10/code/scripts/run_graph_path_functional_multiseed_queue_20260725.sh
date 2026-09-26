#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 QUEUE_ID PHYSICAL_GPU" >&2
  exit 2
fi

queue_id="$1"
physical_gpu="$2"
repo_dir="/data/paperexperiment/LooPlus"
output_root="/data/paperexperiment/graph_path_functional_multiseed_20260725/training"
log_root="/data/paperexperiment/logs/graph_path_functional_multiseed_20260725/training"

case "${queue_id}" in
  q0)
    tasks=(
      "D6_L6 6 6 3"
      "D6_L8 6 8 4"
      "D8_L6 8 6 5"
      "D6_L6 6 6 6"
      "D6_L8 6 8 7"
    )
    ;;
  q1)
    tasks=(
      "D6_L8 6 8 3"
      "D8_L6 8 6 4"
      "D6_L6 6 6 5"
      "D6_L8 6 8 6"
      "D8_L6 8 6 7"
    )
    ;;
  q2)
    tasks=(
      "D8_L6 8 6 3"
      "D6_L6 6 6 4"
      "D6_L8 6 8 5"
      "D8_L6 8 6 6"
      "D6_L6 6 6 7"
    )
    ;;
  *)
    echo "unknown queue: ${queue_id}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_root}" "${log_root}"
cd "${repo_dir}"

for task in "${tasks[@]}"; do
  read -r family max_depth loops seed <<<"${task}"
  run_name="${family}_seed${seed}"
  out_dir="${output_root}/${run_name}"
  log_path="${log_root}/${run_name}.log"
  echo "QUEUE_START $(date --iso-8601=seconds) ${run_name} gpu=${physical_gpu}"
  "${repo_dir}/scripts/train_graph_depth_circuit_seed.sh" \
    "${physical_gpu}" \
    "${max_depth}" \
    "${loops}" \
    "${seed}" \
    "${out_dir}" \
    "${log_path}"
  test -s "${out_dir}/graphpath_N8_D${max_depth}_d256_B2_L${loops}_seed${seed}/best.pt"
  echo "QUEUE_COMPLETE $(date --iso-8601=seconds) ${run_name} gpu=${physical_gpu}"
done
