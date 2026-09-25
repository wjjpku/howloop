#!/usr/bin/env bash
# Run the prespecified two-replica controller stage after *all* G1 locked
# evaluations are available.  Backbones failing the fixed endpoint criterion
# are preserved as endpoint_failed manifests and are not silently dropped.
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
controller_runner="$runner_dir/run_paper2027_graph_g3_controller.sh"
evaluation_runner="$runner_dir/run_paper2027_graph_g3_evaluation.sh"
g2_campaign="$runner_dir/run_paper2027_graph_g2_campaign.sh"
run_root="${PAPER2027_GRAPH_ROOT:-/data/wujiaju/paper2027_confirmatory/graph_g1_v1}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
seeds="${PAPER2027_GRAPH_SEEDS:-100 101 102 103 104 105 106 107 108 109 110 111}"
gpus="${PAPER2027_GRAPH_GPUS:-0 1 2}"
read -r -a seed_array <<< "$seeds"
read -r -a gpu_array <<< "$gpus"
if (( ${#gpu_array[@]} != 3 )); then
  echo "PAPER2027_GRAPH_GPUS must name exactly three physical GPUs" >&2; exit 2
fi

# G2 is controller-free and therefore completes before any G3 fit.  Both use
# the same fixed G1 endpoint-qualification rule and locked test set.
PAPER2027_GRAPH_ROOT="$run_root" PAPER2027_GRAPH_SEEDS="$seeds" PAPER2027_GRAPH_GPUS="$gpus" \
  PAPER2027_PYTHON="$python_bin" bash "$g2_campaign"

for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for slot in 0 1 2; do
    index=$((start + slot)); (( index < ${#seed_array[@]} )) || break
    CUDA_VISIBLE_DEVICES="${gpu_array[slot]}" PAPER2027_GRAPH_SEED="${seed_array[index]}" bash "$controller_runner" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

for ((start=0; start<${#seed_array[@]}; start+=3)); do
  pids=()
  for slot in 0 1 2; do
    index=$((start + slot)); (( index < ${#seed_array[@]} )) || break
    seed="${seed_array[index]}"
    manifest="$run_root/manifests/g3_controller_seed${seed}.json"
    [[ -s "$manifest" ]] || { echo "missing G3 controller manifest: $manifest" >&2; exit 2; }
    status="$("$python_bin" - "$manifest" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
    if [[ "$status" == endpoint_failed ]]; then
      echo "SKIP G3 evaluation seed=$seed endpoint_failed"; continue
    fi
    [[ "$status" == complete ]] || { echo "G3 controller not complete: $manifest ($status)" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${gpu_array[slot]}" PAPER2027_GRAPH_SEED="$seed" bash "$evaluation_runner" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -u "$code_root/scripts/aggregate_paper2027_graph_confirmatory.py" \
  --root "$run_root" --out-dir "$run_root/aggregate_confirmatory" \
  --seeds "${seed_array[@]}" --permutations 512 --bootstrap-draws 10000 --seed 2026096001
