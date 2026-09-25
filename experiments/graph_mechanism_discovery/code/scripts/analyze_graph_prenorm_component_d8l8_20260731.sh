#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu="$1"
repo_dir="${REPO_DIR:-/data/wujiaju/LooPlus_prenorm_component_20260731}"
experiment_root="/data/wujiaju/graph_path_prenorm_component_D8L8_20260731"
natural_root="/data/wujiaju/graph_path_compression_circuit_20260725/training"
component_root="${experiment_root}/training"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"

cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

for seed in 0 1 2 3 4 5; do
  natural_run="${natural_root}/D8_L8_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
  component_run="${component_root}/full/D8_L8_full_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
  test -s "${natural_run}/final.pt"
  test -s "${component_run}/final.pt"
done

"${python_bin}" scripts/summarize_graph_prenorm_component_20260731.py \
  --natural-root "${natural_root}" \
  --component-root "${component_root}" \
  --condition full \
  --seeds 0 1 2 3 4 5 \
  --out-dir "${experiment_root}/training_comparison"

run_args=()
for seed in 0 1 2 3 4 5; do
  natural_checkpoint="${natural_root}/D8_L8_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
  component_checkpoint="${component_root}/full/D8_L8_full_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
  run_args+=(--run "natural_seed${seed}=${natural_checkpoint}")
  run_args+=(--run "full_seed${seed}=${component_checkpoint}")
done
natural_seed1_best="${natural_root}/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt"
component_seed1_step6k="${component_root}/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/checkpoint_step_06000.pt"
test -s "${natural_seed1_best}"
test -s "${component_seed1_step6k}"
run_args+=(--run "natural_seed1_best_step6k=${natural_seed1_best}")
run_args+=(--run "full_seed1_step6k=${component_seed1_step6k}")

"${python_bin}" scripts/analyze_graph_postnorm_clear_circuit_20260730.py \
  "${run_args[@]}" \
  --out-dir "${experiment_root}/macro_analysis" \
  --batch-size 512 \
  --batches 4 \
  --overloops 32 \
  --seed 20260731 \
  --device cuda

"${python_bin}" -m reasoning_loop.graph_path_depth_circuit \
  "${run_args[@]}" \
  --out-dir "${experiment_root}/causal_depth_all_seeds" \
  --batch-size 256 \
  --batches 2 \
  --overloops 16 \
  --seed 20260731 \
  --device cuda

"${python_bin}" scripts/compare_graph_prenorm_component_circuits_20260731.py \
  --analysis-root "${experiment_root}/causal_depth_all_seeds" \
  --macro-root "${experiment_root}/macro_analysis" \
  --seeds 0 1 2 3 4 5 \
  --out-dir "${experiment_root}/circuit_similarity"

seed1_fine_args=(
  --run "natural_seed1_best_step6k=${natural_seed1_best}"
  --run "full_seed1_step6k=${component_seed1_step6k}"
  --run "natural_seed1_final=${natural_root}/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/final.pt"
  --run "full_seed1_final=${component_root}/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/final.pt"
)

"${python_bin}" -m reasoning_loop.graph_path_depth_circuit \
  "${seed1_fine_args[@]}" \
  --out-dir "${experiment_root}/causal_depth_seed1" \
  --batch-size 256 \
  --batches 2 \
  --overloops 16 \
  --seed 20260731 \
  --device cuda

"${python_bin}" -m reasoning_loop.graph_path_functional_circuit \
  "${seed1_fine_args[@]}" \
  --out-dir "${experiment_root}/functional_circuit_seed1" \
  --batch-size 256 \
  --top-k 64 \
  --seed 20260731 \
  --device cuda \
  --force
