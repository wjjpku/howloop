#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU CONFIG SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
config="$2"
shift 2

root="/data/wujiaju/graph_path_hparam_circuit_20260728"
repo="/data/wujiaju/LooPlus"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
out_dir="${root}/behavior_screen/${config}"
log_root="/data/wujiaju/logs/graph_path_hparam_circuit_20260728/behavior_screen"
manifest="${root}/manifests/behavior_${config}.txt"
mkdir -p "${out_dir}" "${log_root}" "$(dirname "${manifest}")"

case "${config}" in
  layers1_b1) n_layers=1 ;;
  layers3_b3) n_layers=3 ;;
  baseline_b2|attn2_mlp05_b2|block2fast_b2|beta2_099_b2|beta1_08_b2)
    n_layers=2
    ;;
  *)
    echo "unknown config: ${config}" >&2
    exit 2
    ;;
esac

run_args=()
for seed in "$@"; do
  checkpoint="${root}/training/${config}/graphpath_N8_D6_d64_B${n_layers}_L6_seed${seed}/final.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  run_args+=(--run "${config}_seed${seed}=${checkpoint}")
done

printf 'status=running\npid=%s\nphysical_gpu=%s\nconfig=%s\nstarted=%s\n' \
  "$$" "${gpu}" "${config}" "$(date --iso-8601=seconds)" >"${manifest}"
cd "${repo}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${repo}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${python_bin}" -u -m reasoning_loop.graph_path_hparam_screen \
  "${run_args[@]}" \
  --out-dir "${out_dir}" \
  --batch-size 1024 \
  --batches 16 \
  --extra-loops 4 \
  --seed 20260728 \
  --device cuda \
  >"${log_root}/${config}.log" 2>&1
printf 'status=complete\npid=%s\nphysical_gpu=%s\nconfig=%s\ncompleted=%s\noutput=%s\n' \
  "$$" "${gpu}" "${config}" "$(date --iso-8601=seconds)" "${out_dir}" \
  >"${manifest}"
