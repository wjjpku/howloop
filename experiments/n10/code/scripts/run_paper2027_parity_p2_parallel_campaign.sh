#!/usr/bin/env bash
# Execute every preregistered P2 controller capacity condition in an isolated
# namespace.  Three GPUs run one label each; no output is shared with an
# interrupted or historical campaign.
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trainer="$runner_dir/run_paper2027_parity_p2_controller.sh"
evaluator="$runner_dir/run_paper2027_parity_p2_evaluation.sh"
run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
seed="${PAPER2027_PARITY_SEED:-5}"
namespace="${PAPER2027_PARITY_P2_NAMESPACE:?set isolated P2 namespace}"
gpus="${PAPER2027_PARITY_P2_GPUS:-0 1 2}"
read -r -a gpu_array <<< "$gpus"
# ``labels`` is execution order, while ``registered_labels`` is the immutable
# order consumed by the aggregate/audit.  Keep them distinct so parallel
# scheduling cannot accidentally alter the protocol declaration.
labels=(rank48_seed1 rank128_seed1 dense_seed1 rank48_seed2 rank128_seed2 dense_seed2)
registered_labels=(rank48_seed1 rank48_seed2 rank128_seed1 rank128_seed2 dense_seed1 dense_seed2)

write_campaign_manifest() {
  local kind="$1" status="$2"
  /data/paperexperiment/.venvs/loopreasoner/bin/python - "$run_root" "$namespace" "$seed" "$kind" "$status" "${registered_labels[@]}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, namespace, seed, kind, status, *labels = sys.argv[1:]
payload = {"status": status, "backbone_seed": int(seed), "namespace": namespace,
           "protocol_id": f"paper2027.parity.p2.{kind}.parallel.v2",
           "labels": labels, "updated_at": datetime.now(timezone.utc).isoformat()}
Path(root, "manifests", f"p2_{kind}_{namespace}_seed{seed}.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
}

write_campaign_manifest controller running

run_batch() {
  local phase="$1"; shift
  local -a pids=()
  local index=0 label gpu
  for label in "$@"; do
    gpu="${gpu_array[$index]}"
    if [[ "$phase" == train ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="$seed" \
        PAPER2027_PARITY_P2_NAMESPACE="$namespace" PAPER2027_PARITY_P2_LABELS="$label" \
        bash "$trainer" &
    else
      local components=0
      [[ "$label" == rank48_seed1 ]] && components=1
      CUDA_VISIBLE_DEVICES="$gpu" PAPER2027_PARITY_SEED="$seed" \
        PAPER2027_PARITY_P2_NAMESPACE="$namespace" PAPER2027_PARITY_P2_LABELS="$label" \
        PAPER2027_PARITY_P2_COMPONENTS="$components" bash "$evaluator" &
    fi
    pids+=("$!")
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

for ((start=0; start<${#labels[@]}; start+=${#gpu_array[@]})); do
  run_batch train "${labels[@]:start:${#gpu_array[@]}}"
done
write_campaign_manifest controller complete
write_campaign_manifest evaluation running
for ((start=0; start<${#labels[@]}; start+=${#gpu_array[@]})); do
  run_batch evaluate "${labels[@]:start:${#gpu_array[@]}}"
done
write_campaign_manifest evaluation complete
