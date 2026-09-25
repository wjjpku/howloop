#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <q|k|v|mlp_pre> <physical_gpu>" >&2
  exit 2
fi

site="$1"
physical_gpu="$2"
case "$site" in
  q|k|v|mlp_pre) ;;
  *) echo "unsupported site: $site" >&2; exit 2 ;;
esac

python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
repo="/data/wujiaju/LooPlus"
backbone="/data/wujiaju/paper_length_telomere_20260731/tn_addition_20260804/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
source_root="/data/wujiaju/paper_length_telomere_20260731/addition_internal_weight_adapter_20260804"
output_root="/data/wujiaju/paper_length_telomere_20260731/addition_internal_weight_adapter_lsb_3x_20260804"
log_root="/data/wujiaju/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_lsb_3x_20260804"
label="lsb_seed0_${site}_total3x"
source_adapter="${source_root}/lsb_seed0_${site}/adapter_best.pt"
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
  --site "$site"
  --out-dir "$out_dir"
  --device cuda
  --seed 0
  --id-min 1
  --id-max 10
  --repair-min 11
  --repair-max 20
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
  --selection-id-lengths 2,5,8,10
  --selection-seen-lengths 11,15,20
  --final-lengths 1,2,5,8,10,11,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
  --eval-batch-size 64
  --selection-eval-batches 2
  --final-eval-batches 8
  --id-gate 0.99
)

{
  printf 'host=%s\n' "$(hostname)"
  printf 'physical_gpu=%s\n' "$physical_gpu"
  printf 'site=%s\n' "$site"
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
