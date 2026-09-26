#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

code_root=/data/paperexperiment/LooPlus_postnorm_20260730
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
checkpoint=/data/paperexperiment/graph_path_postnorm_clear_circuit_20260730/trajectory_w1_hold10k_end15k/graphpath_N8_D8_d256_B2_L8_seed1/checkpoint_step_14000.pt
experiment_root=/data/paperexperiment/postnorm_tangent_rejuvenator_20260730
output_root="${experiment_root}/strict_reeval"
log_root=/data/paperexperiment/logs/postnorm_tangent_rejuvenator_20260730

mkdir -p "${output_root}" "${log_root}"
cd "${code_root}"

run_one() {
  local name=$1
  local artifact=$2
  echo "START ${name} $(date -Is)" >> "${log_root}/strict_reeval_master.log"
  "${python_bin}" -m scripts.reevaluate_graph_path_postnorm_tangent_strict_20260731 \
    --checkpoint "${checkpoint}" \
    --artifact "${artifact}" \
    --out-csv "${output_root}/${name}.csv" \
    --batch-size 512 \
    --batches 4 \
    --cycles 200 \
    --seed 9732 \
    --device cuda \
    > "${log_root}/strict_${name}.log" 2>&1
  echo "DONE ${name} $(date -Is)" >> "${log_root}/strict_reeval_master.log"
}

run_one tangent_adjacent_answer "${experiment_root}/formal_screen/tangent_adjacent_answer/rejuvenator.pt"
run_one tangent_terminal_reset_all "${experiment_root}/formal_closed_loop/tangent_terminal_reset_all/rejuvenator.pt"
run_one euclidean_terminal_reset_all "${experiment_root}/formal_closed_loop/euclidean_terminal_reset_all/rejuvenator.pt"
run_one tangent_direct_all_functional "${experiment_root}/formal_closed_loop/tangent_direct_all_functional/rejuvenator.pt"
run_one tangent_rollout4 "${experiment_root}/formal_rollout/tangent_rollout4/rejuvenator.pt"
run_one euclidean_rollout4 "${experiment_root}/formal_rollout/euclidean_rollout4/rejuvenator.pt"
run_one tangent_rollout8 "${experiment_root}/formal_rollout/tangent_rollout8/rejuvenator.pt"
echo "ALL_DONE $(date -Is)" >> "${log_root}/strict_reeval_master.log"
