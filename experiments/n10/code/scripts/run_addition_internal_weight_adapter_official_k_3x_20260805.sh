#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical_gpu>" >&2
  exit 2
fi

physical_gpu="$1"
python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"
repo="/data/paperexperiment/LooPlus"
backbone="/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_adaptive_step_official_seed0/checkpoint_100000.pt"
source_adapter="/data/paperexperiment/paper_length_telomere_20260731/addition_internal_weight_adapter_20260804/official_seed0_k/adapter_best.pt"
output_root="/data/paperexperiment/paper_length_telomere_20260731/addition_internal_weight_adapter_official_k_3x_20260805"
log_root="/data/paperexperiment/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_official_k_3x_20260805"
label="official_seed0_k_total3x"
out_dir="${output_root}/${label}"
log_dir="${log_root}/${label}"

if [[ ! -f "$source_adapter" ]]; then
  echo "missing source adapter: $source_adapter" >&2
  exit 1
fi
if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty output directory: $out_dir" >&2
  exit 1
fi
mkdir -p "$log_dir"

command=(
  "$python_bin" -m reasoning_loop.addition_internal_weight_adapter
  --checkpoint "$backbone"
  --adapter-checkpoint "$source_adapter"
  --prior-updates 5376
  --site k
  --out-dir "$out_dir"
  --device cuda
  --seed 0
  --id-min 1
  --id-max 19
  --repair-min 20
  --repair-max 40
  --updates 10752
  --batch-size 16
  --learning-rate 2e-5
  --init-std 1e-4
  --warmup-updates 256
  --stable-updates 9472
  --final-lr-ratio 0.1
  --grad-clip 1.0
  --gradient-checkpointing
  --amp
  --eval-every 256
  --selection-id-lengths 5,10,15,19
  --selection-seen-lengths 20,30,40
  --final-lengths 1,5,10,15,19,20,25,30,35,40,41,45,50,55,60,65,70,75,80,90,100
  --eval-batch-size 64
  --selection-eval-batches 2
  --final-eval-batches 8
  --id-gate 0.99
)

{
  printf 'host=%s\n' "$(hostname)"
  printf 'physical_gpu=%s\n' "$physical_gpu"
  printf 'baseline=official\n'
  printf 'site=k\n'
  printf 'source_adapter=%s\n' "$source_adapter"
  printf 'prior_updates=5376\n'
  printf 'continuation_updates=10752\n'
  printf 'cumulative_updates=16128\n'
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "${log_dir}/launch.txt"

cd "$repo"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
exec "${command[@]}"
