#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU CONDITION SEED [SEED ...]" >&2
  echo "CONDITION must be full, active, or onehop" >&2
  exit 2
fi

physical_gpu="$1"
condition="$2"
shift 2

if [[ "${condition}" != "full" && "${condition}" != "active" && "${condition}" != "onehop" ]]; then
  echo "CONDITION must be full, active, or onehop" >&2
  exit 2
fi

repo_dir="${REPO_DIR:-/data/wujiaju/LooPlus_prenorm_component_20260731}"
experiment_root="/data/wujiaju/graph_path_prenorm_component_D8L8_20260731"
out_root="${experiment_root}/training/${condition}"
log_root="/data/wujiaju/logs/graph_path_prenorm_component_D8L8_20260731/${condition}"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"

mkdir -p "${out_root}" "${log_root}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

active_flag="--no-trajectory-aux-active-only"
jump=2
if [[ "${condition}" == "active" ]]; then
  active_flag="--trajectory-aux-active-only"
elif [[ "${condition}" == "onehop" ]]; then
  jump=1
fi

for seed in "$@"; do
  run_out="${out_root}/D8_L8_${condition}_seed${seed}"
  run_dir="${run_out}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
  run_log="${log_root}/D8_L8_${condition}_seed${seed}.log"
  checkpoint="${run_dir}/final.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "SKIP condition=${condition} seed=${seed} checkpoint=${checkpoint}"
    continue
  fi

  echo "START $(date --iso-8601=seconds) condition=${condition} seed=${seed} gpu=${physical_gpu}" \
    >"${run_log}"
  "${python_bin}" -u -m reasoning_loop.graph_path_loop \
    --node-count 8 \
    --max-depth 8 \
    --d-model 256 \
    --n-heads 4 \
    --d-mlp 1024 \
    --n-layers 2 \
    --loops 8 \
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
    --dropout 0.0 \
    --inner-norm-style pre_layernorm \
    --aux-loss 0.0 \
    --trajectory-aux-weight 1.0 \
    --trajectory-aux-jump "${jump}" \
    "${active_flag}" \
    --trajectory-aux-hold-steps 0 \
    --trajectory-aux-end-step 0 \
    --device cuda \
    --amp \
    --no-compile \
    --save-checkpoints \
    --save-eval-checkpoints \
    --out-dir "${run_out}" \
    >>"${run_log}" 2>&1
  test -s "${checkpoint}"
  echo "COMPLETE $(date --iso-8601=seconds) condition=${condition} seed=${seed} gpu=${physical_gpu}" \
    >>"${run_log}"
done
