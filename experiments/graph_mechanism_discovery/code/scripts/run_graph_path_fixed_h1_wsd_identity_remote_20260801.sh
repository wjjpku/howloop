#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set one physical GPU}"
if (( $# != 1 )); then
  echo "usage: $0 SHARED_RANK" >&2
  exit 2
fi

shared_rank=$1
stage_rank=16
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
repo_dir=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase_summary=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
result_root=/data/wujiaju/graph_path_fixed_h1_j_20260801/wsd_identity_2x/r${shared_rank}_s${stage_rank}
log_root=/data/wujiaju/logs/graph_path_fixed_h1_j_20260801
mkdir -p "$result_root" "$log_root"
cd "$repo_dir"

common_args=(
  --checkpoint "$checkpoint"
  --phase-summary "$phase_summary"
  --rank "$shared_rank"
  --stage-rank "$stage_rank"
  --map-architecture shared_diagonal_stage_lora
  --rollback-composition product
  --fixed-start-age 1
  --lr-schedule wsd
  --warmup-fraction 0.2
  --warmup-start-factor 0.1
  --decay-fraction 0.2
  --decay-end-factor 0.1
  --stage-round-multiplier 2
  --diagonal-lr-multiplier 1.0
  --cuda-memory-fraction 0.02
  --seed 820001
)

single_dir="$result_root/single_h1"
focused_dir="$result_root/focused5_h1"
single_artifact="$single_dir/age_specific_j_bank_after_T01_R1.pt"
focused_artifact="$focused_dir/age_specific_j_bank_after_T05_R5.pt"

if [[ ! -s "$single_dir/summary.json" || ! -s "$single_artifact" ]]; then
  "$python_bin" -u -m reasoning_loop.train_graph_path_age_specific_j_bank \
    "${common_args[@]}" \
    --out-dir "$single_dir" \
    --initialization identity \
    --curriculum single_back \
    >"$log_root/wsd_identity_2x_r${shared_rank}_s${stage_rank}_single_gpu${CUDA_VISIBLE_DEVICES}.log" 2>&1
fi

if [[ ! -s "$focused_dir/summary.json" || ! -s "$focused_artifact" ]]; then
  "$python_bin" -u -m reasoning_loop.train_graph_path_age_specific_j_bank \
    "${common_args[@]}" \
    --out-dir "$focused_dir" \
    --initialization age_specific_bank \
    --bank-init-artifact "$single_artifact" \
    --curriculum focused5 \
    >"$log_root/wsd_identity_2x_r${shared_rank}_s${stage_rank}_focused_gpu${CUDA_VISIBLE_DEVICES}.log" 2>&1
fi

"$python_bin" - "$single_dir/summary.json" "$focused_dir/summary.json" "$focused_artifact" <<'PY'
import json
import pathlib
import sys

single = json.loads(pathlib.Path(sys.argv[1]).read_text())
focused = json.loads(pathlib.Path(sys.argv[2]).read_text())
artifact = pathlib.Path(sys.argv[3])
assert single["status"] == focused["status"] == "complete"
assert single["training_start_age"] == focused["training_start_age"] == 1
assert focused["lr_schedule"]["name"] == "wsd"
assert focused["lr_schedule"]["diagonal_lr_multiplier"] == 1.0
assert focused["lr_schedule"]["stage_round_multiplier"] == 2
assert artifact.is_file() and artifact.stat().st_size > 0
print(f"validated {artifact}")
PY

