#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU SEED [SEED ...]" >&2
  exit 2
fi

physical_gpu="$1"
shift

repo_dir="${REPO_DIR:-/data/wujiaju/LooPlus_postnorm_20260730}"
out_root="/data/wujiaju/graph_path_postnorm_D8L8_20260730/training"
log_root="/data/wujiaju/logs/graph_path_postnorm_D8L8_20260730/training"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"

mkdir -p "${out_root}" "${log_root}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in "$@"; do
  run_out="${out_root}/D8_L8_postnorm_seed${seed}"
  run_dir="${run_out}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
  run_log="${log_root}/D8_L8_postnorm_seed${seed}.log"
  checkpoint="${run_dir}/final.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "SKIP seed=${seed} checkpoint=${checkpoint}"
    continue
  fi
  echo "START $(date --iso-8601=seconds) seed=${seed} gpu=${physical_gpu}" \
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
    --warmup-steps 2000 \
    --grad-clip 1.0 \
    --seed "${seed}" \
    --dropout 0.0 \
    --inner-norm-style post_layernorm \
    --aux-loss 0.0 \
    --device cuda \
    --amp \
    --no-compile \
    --save-checkpoints \
    --save-eval-checkpoints \
    --out-dir "${run_out}" \
    >>"${run_log}" 2>&1
  test -s "${checkpoint}"
  echo "COMPLETE $(date --iso-8601=seconds) seed=${seed} gpu=${physical_gpu}" \
    >>"${run_log}"
done
