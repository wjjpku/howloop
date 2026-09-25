#!/usr/bin/env bash
# Re-evaluate every registered G4 backbone after an evaluator-only failure.
# This never overwrites a failed tree: the new evaluation directory and its
# manifests are part of the retry provenance, while backbones/controllers are
# immutable inputs.
set -euo pipefail

run_root="${PAPER2027_GRAPH_G4_ROOT:?set PAPER2027_GRAPH_G4_ROOT}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$runner_dir/run_paper2027_graph_g4_evaluation.sh"
evaluation_dir="${PAPER2027_GRAPH_G4_RETRY_EVALUATION_DIR:-g4_evaluation_rerun_v1}"
manifest_prefix="${PAPER2027_GRAPH_G4_RETRY_MANIFEST_PREFIX:-evaluation_rerun_v1_seed}"
seeds="${PAPER2027_GRAPH_G4_SEEDS:-100 101 102 103 104 105 106 107 108 109 110 111}"
gpus="${PAPER2027_GRAPH_G4_RETRY_GPUS:-0 1 2}"

[[ -s "$runner" ]] || { echo "missing evaluator runner: $runner" >&2; exit 2; }
[[ "$evaluation_dir" != */* && "$evaluation_dir" != .* && "$evaluation_dir" != "" ]] || { echo "invalid retry evaluation directory" >&2; exit 2; }
[[ "$manifest_prefix" != */* && "$manifest_prefix" != .* && "$manifest_prefix" != "" ]] || { echo "invalid retry manifest prefix" >&2; exit 2; }
if [[ -e "$run_root/$evaluation_dir" ]]; then
  echo "refusing to overwrite retry evaluation directory: $run_root/$evaluation_dir" >&2
  exit 2
fi
for seed in $seeds; do
  manifest="$run_root/manifests/controller_seed${seed}.json"
  [[ -s "$manifest" ]] || { echo "missing controller manifest seed=$seed" >&2; exit 2; }
  "$python_bin" - "$manifest" "$seed" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "complete":
    raise SystemExit(f"controller seed {sys.argv[2]} is not complete")
PY
done

read -r -a seed_array <<< "$seeds"
read -r -a gpu_array <<< "$gpus"
(( ${#gpu_array[@]} > 0 )) || { echo "need at least one retry GPU" >&2; exit 2; }
for ((start=0; start<${#seed_array[@]}; start+=${#gpu_array[@]})); do
  pids=()
  for gpu_index in "${!gpu_array[@]}"; do
    gpu="${gpu_array[gpu_index]}"
    index=$((start + gpu_index))
    (( index < ${#seed_array[@]} )) || break
    seed="${seed_array[index]}"
    CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_GRAPH_SEED="$seed" \
      PAPER2027_GRAPH_G4_ROOT="$run_root" PAPER2027_PYTHON="$python_bin" \
      PAPER2027_GRAPH_G4_EVALUATION_DIR="$evaluation_dir" \
      PAPER2027_GRAPH_G4_EVALUATION_MANIFEST_PREFIX="$manifest_prefix" \
      bash "$runner" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

echo "G4 retry evaluation complete: $evaluation_dir"
