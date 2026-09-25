#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical_gpu>" >&2
  exit 2
fi

physical_gpu="$1"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
repo="/data/wujiaju/LooPlus"
run_root="/data/wujiaju/paper_length_telomere_20260731"
train_dir="${run_root}/addition_internal_weight_adapter_official_k_3x_20260805/official_seed0_k_total3x"
train_log_dir="/data/wujiaju/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_official_k_3x_20260805/official_seed0_k_total3x"
comparison_root="${run_root}/addition_internal_weight_adapter_official_k_3x_20260805/comparison"
log_root="/data/wujiaju/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_official_k_3x_20260805/comparison"
checkpoint="${run_root}/backbones/addition_adaptive_step_official_seed0/checkpoint_100000.pt"
k_1x="${run_root}/addition_internal_weight_adapter_20260804/official_seed0_k/adapter_best.pt"
k_2x="${train_dir}/checkpoints/adapter_005376.pt"
k_3x="${train_dir}/adapter_best.pt"
j_controller="${run_root}/controllers/addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_postfinal_wsd_lr1em4_warmup2048_stable2816_5k_seed211001/controller.pt"

mkdir -p "$log_root"
while true; do
  train_exit=""
  if [[ -f "${train_log_dir}/exit_code" ]]; then
    train_exit="$(tr -d '[:space:]' < "${train_log_dir}/exit_code")"
    if [[ "$train_exit" != "0" ]]; then
      echo "training failed with exit code ${train_exit}" >&2
      exit 1
    fi
  fi
  if [[ "$train_exit" == "0" ]] && [[ -f "${train_dir}/summary.json" ]] && grep -q '"status": "complete"' "${train_dir}/summary.json"; then
    break
  fi
  sleep 30
done

if [[ ! -f "$k_2x" ]] || [[ ! -f "$k_3x" ]]; then
  echo "missing completed K budget checkpoint: $k_2x or $k_3x" >&2
  exit 1
fi

run_eval() {
  local label="$1"
  shift
  local out_dir="${comparison_root}/${label}"
  if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing non-empty evaluation directory: $out_dir" >&2
    exit 1
  fi
  "$python_bin" -m reasoning_loop.evaluate_addition_internal_adapter_vs_j \
    --checkpoint "$checkpoint" \
    --adapter "k_1x=${k_1x}" \
    --adapter "k_2x=${k_2x}" \
    --adapter "k_3x=${k_3x}" \
    --controller "$j_controller" \
    --device cuda \
    --batch-size 32 \
    --out-dir "$out_dir" \
    "$@"
}

cd "$repo"
export CUDA_VISIBLE_DEVICES="$physical_gpu"

run_eval dense_l1to100_n128 \
  --max-length 100 \
  --evaluation-seeds 571001,581001,591001,601001 \
  > "${log_root}/dense_l1to100_n128.log" 2>&1

run_eval anchors_n512 \
  --lengths 1,5,10,15,19,20,25,30,35,40,41,45,50,55,60,65,70,75,80,90,100 \
  --evaluation-seeds 571001,581001,591001,601001,611001,621001,631001,641001,651001,661001,671001,681001,691001,701001,711001,721001 \
  > "${log_root}/anchors_n512.log" 2>&1

printf '0\n' > "${log_root}/exit_code"
