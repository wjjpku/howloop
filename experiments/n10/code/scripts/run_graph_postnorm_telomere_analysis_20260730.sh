#!/usr/bin/env bash
set -euo pipefail

code_root=/data/paperexperiment/LooPlus_postnorm_20260730
experiment_root=/data/paperexperiment/graph_path_postnorm_telomere_20260730
log_path=/data/paperexperiment/logs/graph_path_postnorm_telomere_formal_20260730.log
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python

post_trajectory_checkpoint=/data/paperexperiment/graph_path_postnorm_clear_circuit_20260730/trajectory_w1_hold10k_end15k/graphpath_N8_D8_d256_B2_L8_seed1/checkpoint_step_14000.pt
post_final_checkpoint=/data/paperexperiment/graph_path_postnorm_clear_circuit_20260730/final_only_warmup5k/graphpath_N8_D8_d256_B2_L8_seed1/final.pt
pre_final_checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt

mkdir -p "${experiment_root}" "$(dirname "${log_path}")"
cd "${code_root}"

for checkpoint in \
  "${post_trajectory_checkpoint}" \
  "${post_final_checkpoint}" \
  "${pre_final_checkpoint}"
do
  test -f "${checkpoint}"
done

printf '{"status":"running","physical_gpu":2,"expected_peak_gib":2,"output":"%s","log":"%s"}\n' \
  "${experiment_root}" "${log_path}" > "${experiment_root}/run_manifest.json"

{
  echo "started $(date --iso-8601=seconds)"
  echo "physical_gpu=2"
  echo "dynamics_output=${experiment_root}/formal_dynamics"
  echo "causal_output=${experiment_root}/formal_causal"
} | tee "${log_path}"

CUDA_VISIBLE_DEVICES=2 "${python_bin}" \
  -m scripts.analyze_graph_postnorm_telomere_dynamics_20260730 \
  --run "post_traj_seed1=${post_trajectory_checkpoint}" \
  --run "post_final_seed1=${post_final_checkpoint}" \
  --run "pre_final_seed1=${pre_final_checkpoint}" \
  --out-dir "${experiment_root}/formal_dynamics" \
  --device cuda \
  --batch-size 256 \
  --batches 4 \
  --loops 200 \
  --probe-batch-size 256 \
  --probe-batches 2 \
  --ridge 1e-3 \
  --seed 20260730 2>&1 | tee -a "${log_path}"

CUDA_VISIBLE_DEVICES=2 "${python_bin}" \
  -m reasoning_loop.graph_path_telomere_overloop \
  --run "post_traj_seed1=${post_trajectory_checkpoint}" \
  --run "post_final_seed1=${post_final_checkpoint}" \
  --run "pre_final_seed1=${pre_final_checkpoint}" \
  --objective-label "post_traj_seed1=final_plus_trajectory_aux_lambda0p2" \
  --objective-label "post_final_seed1=pure_final_only" \
  --objective-label "pre_final_seed1=pure_final_only" \
  --out-dir "${experiment_root}/formal_causal" \
  --calibration-batch-size 256 \
  --calibration-batches 4 \
  --selection-batch-size 512 \
  --evaluation-batch-size 256 \
  --evaluation-batches 4 \
  --component-batch-size 128 \
  --extra-loops 4 \
  --seed 20260730 \
  --device cuda 2>&1 | tee -a "${log_path}"

printf '{"status":"complete","physical_gpu":2,"expected_peak_gib":2,"output":"%s","log":"%s"}\n' \
  "${experiment_root}" "${log_path}" > "${experiment_root}/run_manifest.json"
touch "${experiment_root}/FORMAL_COMPLETE"
echo "completed $(date --iso-8601=seconds)" | tee -a "${log_path}"
