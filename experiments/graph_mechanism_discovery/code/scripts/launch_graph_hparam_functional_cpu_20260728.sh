#!/usr/bin/env bash
set -euo pipefail

root="/data/wujiaju/graph_path_hparam_circuit_20260728"
repo="/data/wujiaju/LooPlus"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
log_root="/data/wujiaju/logs/graph_path_hparam_circuit_20260728/functional_cpu"
mkdir -p "${root}/functional_raw" "${root}/manifests" "${log_root}"

if [[ $# -eq 0 ]]; then
  set -- \
    baseline_b2 \
    attn2_mlp05_b2 \
    block2fast_b2 \
    beta2_099_b2 \
    beta1_08_b2 \
    layers1_b1 \
    layers3_b3
fi

run_config() {
  local config="$1"
  local n_layers
  local out_dir="${root}/functional_raw/${config}"
  local manifest="${root}/manifests/functional_${config}.txt"
  local run_args=()

  case "${config}" in
    layers1_b1) n_layers=1 ;;
    layers3_b3) n_layers=3 ;;
    baseline_b2|attn2_mlp05_b2|block2fast_b2|beta2_099_b2|beta1_08_b2)
      n_layers=2
      ;;
    *)
      echo "unknown config: ${config}" >&2
      return 2
      ;;
  esac

  for seed in 0 1 2 3 4; do
    local checkpoint="${root}/training/${config}/graphpath_N8_D6_d64_B${n_layers}_L6_seed${seed}/final.pt"
    if [[ ! -s "${checkpoint}" ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      return 1
    fi
    run_args+=(--run "${config}_seed${seed}=${checkpoint}")
  done

  printf 'status=running\npid=%s\ndevice=cpu\nconfig=%s\nstarted=%s\n' \
    "${BASHPID}" "${config}" "$(date --iso-8601=seconds)" >"${manifest}"
  (
    cd "${repo}"
    export PYTHONPATH="${repo}"
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    "${python_bin}" -u -m reasoning_loop.graph_path_functional_circuit \
      "${run_args[@]}" \
      --out-dir "${out_dir}" \
      --batch-size 512 \
      --top-k 16 \
      --seed 20260728 \
      --device cpu \
      --force
  ) >"${log_root}/${config}.log" 2>&1
  printf 'status=complete\npid=%s\ndevice=cpu\nconfig=%s\ncompleted=%s\noutput=%s\n' \
    "${BASHPID}" "${config}" "$(date --iso-8601=seconds)" "${out_dir}" \
    >"${manifest}"
}

pids=()
for config in "$@"; do
  run_config "${config}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
