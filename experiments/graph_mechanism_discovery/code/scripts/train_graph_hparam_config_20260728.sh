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
out_root="${root}/training/${config}"
log_root="/data/wujiaju/logs/graph_path_hparam_circuit_20260728/training/${config}"
manifest_suffix="${GRAPH_HPARAM_MANIFEST_SUFFIX:-}"
manifest="${root}/manifests/training_${config}${manifest_suffix}.txt"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
repo="/data/wujiaju/LooPlus"
mkdir -p "${out_root}" "${log_root}" "$(dirname "${manifest}")"

n_layers=2
extra_args=()
case "${config}" in
  baseline_b2)
    ;;
  attn2_mlp05_b2)
    extra_args+=(--attention-lr-scale 2.0 --mlp-lr-scale 0.5)
    ;;
  block2fast_b2)
    extra_args+=(--block-lr-scales 0.5 1.5)
    ;;
  beta2_099_b2)
    extra_args+=(--adam-beta2 0.99)
    ;;
  beta1_08_b2)
    extra_args+=(--adam-beta1 0.8)
    ;;
  layers1_b1)
    n_layers=1
    ;;
  layers3_b3)
    n_layers=3
    ;;
  *)
    echo "unknown config: ${config}" >&2
    exit 2
    ;;
esac

status="failed"
finish_manifest() {
  printf 'status=%s\npid=%s\nphysical_gpu=%s\nconfig=%s\nfinished=%s\noutput=%s\n' \
    "${status}" "$$" "${gpu}" "${config}" "$(date --iso-8601=seconds)" \
    "${out_root}" >"${manifest}"
}
trap finish_manifest EXIT
printf 'status=running\npid=%s\nphysical_gpu=%s\nconfig=%s\nstarted=%s\noutput=%s\n' \
  "$$" "${gpu}" "${config}" "$(date --iso-8601=seconds)" "${out_root}" \
  >"${manifest}"

cd "${repo}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}"

for seed in "$@"; do
  initialization_seed="$((2026072800 + seed))"
  data_seed="$((2026072900 + seed))"
  run_dir="${out_root}/graphpath_N8_D6_d64_B${n_layers}_L6_seed${seed}"
  checkpoint="${run_dir}/final.pt"
  run_log="${log_root}/seed${seed}.log"
  if [[ -s "${checkpoint}" ]]; then
    echo "SKIP config=${config} seed=${seed} checkpoint=${checkpoint}"
    continue
  fi
  if [[ -d "${run_dir}" ]]; then
    echo "WAIT config=${config} seed=${seed} partial_run=${run_dir}"
    for _ in $(seq 1 180); do
      sleep 20
      if [[ -s "${checkpoint}" ]]; then
        echo "SKIP_AFTER_WAIT config=${config} seed=${seed} checkpoint=${checkpoint}"
        break
      fi
    done
    if [[ -s "${checkpoint}" ]]; then
      continue
    fi
    echo "partial run did not finish within 60 minutes: ${run_dir}" >&2
    exit 1
  fi
  {
    echo "START $(date --iso-8601=seconds)"
    echo "PHYSICAL_GPU ${gpu}"
    echo "CONFIG ${config}"
    echo "INITIALIZATION_SEED ${initialization_seed}"
    echo "DATA_SEED ${data_seed}"
    "${python_bin}" -u -m reasoning_loop.graph_path_loop \
      --node-count 8 \
      --max-depth 6 \
      --d-model 64 \
      --n-heads 4 \
      --d-mlp 256 \
      --n-layers "${n_layers}" \
      --loops 6 \
      --steps 20000 \
      --batch-size 512 \
      --eval-batch-size 1024 \
      --eval-batches 16 \
      --eval-every 1000 \
      --print-every 1000 \
      --lr 0.0003 \
      --weight-decay 0.3 \
      --warmup-steps 500 \
      --grad-clip 1.0 \
      --seed "${seed}" \
      --initialization-seed "${initialization_seed}" \
      --data-seed "${data_seed}" \
      --dropout 0.0 \
      --aux-loss 0.0 \
      --device cuda \
      --amp \
      --no-compile \
      --save-checkpoints \
      --out-dir "${out_root}" \
      "${extra_args[@]}"
    echo "COMPLETE $(date --iso-8601=seconds)"
  } >"${run_log}" 2>&1
done

status="complete"
