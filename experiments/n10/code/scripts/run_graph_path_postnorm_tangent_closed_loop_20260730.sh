#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

code_root=/data/wujiaju/LooPlus_postnorm_20260730
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
checkpoint=/data/wujiaju/graph_path_postnorm_clear_circuit_20260730/trajectory_w1_hold10k_end15k/graphpath_N8_D8_d256_B2_L8_seed1/checkpoint_step_14000.pt
output_root=/data/wujiaju/postnorm_tangent_rejuvenator_20260730/formal_closed_loop
log_root=/data/wujiaju/logs/postnorm_tangent_rejuvenator_20260730

mkdir -p "${output_root}" "${log_root}"
cd "${code_root}"

run_one() {
  local name=$1
  local transform=$2
  local positions=$3
  local pairs=$4
  echo "START ${name} $(date -Is)" >> "${log_root}/formal_closed_loop_master.log"
  "${python_bin}" -m reasoning_loop.graph_path_postnorm_tangent_rejuvenator \
    --checkpoint "${checkpoint}" \
    --out-dir "${output_root}/${name}" \
    --transform "${transform}" \
    --position-mode "${positions}" \
    --pair-mode "${pairs}" \
    --steps 2000 \
    --batch-size 256 \
    --lr 0.0005 \
    --state-weight 0.1 \
    --post-state-weight 0.05 \
    --current-ce-weight 1.0 \
    --next-ce-weight 5.0 \
    --eval-batch-size 512 \
    --eval-batches 4 \
    --eval-cycles 200 \
    --print-every 100 \
    --device cuda \
    > "${log_root}/${name}.log" 2>&1
  echo "DONE ${name} $(date -Is)" >> "${log_root}/formal_closed_loop_master.log"
}

run_one tangent_terminal_reset_all tangent all terminal_reset
run_one tangent_terminal_reset_answer tangent answer terminal_reset
run_one euclidean_terminal_reset_all euclidean all terminal_reset
run_one tangent_direct_all_functional tangent all direct
echo "ALL_DONE $(date -Is)" >> "${log_root}/formal_closed_loop_master.log"
