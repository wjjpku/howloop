#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <official|lsb> <q|k|v|mlp_pre> <physical_gpu>" >&2
  exit 2
fi

baseline="$1"
site="$2"
physical_gpu="$3"

case "$site" in
  q|k|v|mlp_pre) ;;
  *) echo "unsupported site: $site" >&2; exit 2 ;;
esac

case "$baseline" in
  official)
    checkpoint="/data/paperexperiment/paper_length_telomere_20260731/backbones/addition_adaptive_step_official_seed0/checkpoint_100000.pt"
    id_min=1
    id_max=19
    repair_min=20
    repair_max=40
    selection_id="5,10,15,19"
    selection_seen="20,30,40"
    final_lengths="1,5,10,15,19,20,25,30,35,40,41,45,50,55,60"
    ;;
  lsb)
    checkpoint="/data/paperexperiment/paper_length_telomere_20260731/tn_addition_20260804/backbones/addition_lsb_variable_m1to10_tn_logicaldigits_nope_seed0/checkpoint_080000.pt"
    id_min=1
    id_max=10
    repair_min=11
    repair_max=20
    selection_id="2,5,8,10"
    selection_seen="11,15,20"
    final_lengths="1,2,5,8,10,11,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30"
    ;;
  *) echo "unsupported baseline: $baseline" >&2; exit 2 ;;
esac

python_bin="/data/paperexperiment/.venvs/loopreasoner/bin/python"
repo="/data/paperexperiment/LooPlus"
output_root="/data/paperexperiment/paper_length_telomere_20260731/addition_internal_weight_adapter_20260804"
log_root="/data/paperexperiment/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_20260804"
label="${baseline}_seed0_${site}"
out_dir="${output_root}/${label}"
log_dir="${log_root}/${label}"

if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty output directory: $out_dir" >&2
  exit 1
fi
mkdir -p "$log_dir"

command=(
  "$python_bin" -m reasoning_loop.addition_internal_weight_adapter
  --checkpoint "$checkpoint"
  --site "$site"
  --out-dir "$out_dir"
  --device cuda
  --seed 0
  --id-min "$id_min"
  --id-max "$id_max"
  --repair-min "$repair_min"
  --repair-max "$repair_max"
  --updates 5376
  --batch-size 16
  --learning-rate 1e-4
  --init-std 1e-4
  --warmup-updates 2048
  --stable-updates 2816
  --final-lr-ratio 0.1
  --grad-clip 1.0
  --gradient-checkpointing
  --amp
  --eval-every 256
  --selection-id-lengths "$selection_id"
  --selection-seen-lengths "$selection_seen"
  --final-lengths "$final_lengths"
  --eval-batch-size 64
  --selection-eval-batches 2
  --final-eval-batches 8
  --id-gate 0.99
)

{
  printf 'host=%s\n' "$(hostname)"
  printf 'physical_gpu=%s\n' "$physical_gpu"
  printf 'baseline=%s\n' "$baseline"
  printf 'site=%s\n' "$site"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "${log_dir}/launch.txt"

cd "$repo"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
exec "${command[@]}"
