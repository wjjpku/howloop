#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU SEED [SEED ...]" >&2
  exit 2
fi

gpu="$1"
shift
manifest_seed="$1"

root="/data/wujiaju/graph_path_induction_contrast_20260726"
out_root="${root}/training_stretch"
log_root="/data/wujiaju/logs/graph_path_induction_contrast_20260726/training_stretch"
manifest="${root}/training_stretch_manifest_seed${manifest_seed}.txt"
mkdir -p "${out_root}" "${log_root}"
printf 'status=running\npid=%s\nphysical_gpu=%s\nstarted=%s\noutput=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" "${out_root}" >"${manifest}"

for seed in "$@"; do
  initialization_seed="$((202607360 + seed))"
  data_seed="$((202608100 + seed))"
  for loops in 6 8; do
    run_out="${out_root}/D6_L${loops}_pairseed${seed}"
    run_log="${log_root}/D6_L${loops}_pairseed${seed}.log"
    checkpoint="${run_out}/graphpath_N8_D6_d256_B2_L${loops}_seed${seed}/final.pt"
    if [[ -s "${checkpoint}" ]]; then
      echo "SKIP pairseed=${seed} loops=${loops} checkpoint=${checkpoint}"
      continue
    fi
    mkdir -p "${run_out}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    {
      echo "START $(date --iso-8601=seconds)"
      echo "PHYSICAL_GPU ${gpu}"
      echo "INITIALIZATION_SEED ${initialization_seed}"
      echo "DATA_SEED ${data_seed}"
      /data/wujiaju/.venvs/loopreasoner/bin/python -u \
        -m reasoning_loop.graph_path_loop \
        --node-count 8 \
        --max-depth 6 \
        --d-model 256 \
        --n-heads 4 \
        --d-mlp 1024 \
        --n-layers 2 \
        --loops "${loops}" \
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
        --out-dir "${run_out}"
      echo "COMPLETE $(date --iso-8601=seconds)"
    } >"${run_log}" 2>&1
  done
done
printf 'status=complete\npid=%s\nphysical_gpu=%s\ncompleted=%s\noutput=%s\n' \
  "$$" "${gpu}" "$(date --iso-8601=seconds)" "${out_root}" >"${manifest}"
