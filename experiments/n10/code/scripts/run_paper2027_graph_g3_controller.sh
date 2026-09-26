#!/usr/bin/env bash
# Pure-CE G3 controller fits for one independently trained G1 backbone.
# Qualification uses only call-8 endpoint accuracy from the already locked G1
# test; no post-horizon controller result can select a backbone.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g1_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
trainer="$code_root/reasoning_loop/paper2027_graph_g3_controller.py"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
g1_summary="$run_root/evaluation/seed${seed}/summary.json"
controller_root="$run_root/g3_controllers/seed${seed}"
log_dir="/data/paperexperiment/logs/paper2027_confirmatory/graph_g1_v1"
log_file="$log_dir/g3_controller_seed${seed}.log"
manifest="$run_root/manifests/g3_controller_seed${seed}.json"

mkdir -p "$controller_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$g1_summary" && -s "$trainer" ]] || {
  echo "G3 needs final G1 checkpoint, locked G1 evaluation, and analysis source" >&2; exit 2;
}

qualification="$("$python_bin" - "$g1_summary" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
row = next((r for r in summary["aggregate_rows"] if int(r["call"]) == 8), None)
if row is None: raise SystemExit("locked G1 summary has no call-8 row")
accuracy = float(row["endpoint_hold_accuracy"])
print("qualified" if accuracy >= .95 else "endpoint_failed", f"{accuracy:.12g}")
PY
)"
read -r selection_status call8_accuracy <<< "$qualification"

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$g1_summary" "$trainer" "$log_file" "$selection_status" "$call8_accuracy" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, seed, checkpoint, g1, source, log, selection, call8 = sys.argv[1:]
payload = {
 "status":status, "updated_at":datetime.now(timezone.utc).isoformat(),
 "backbone_seed":int(seed), "protocol_id":"paper2027.graph.g3.controller.v1",
 "checkpoint":checkpoint, "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
 "g1_locked_summary":g1, "g1_locked_summary_sha256":hashlib.sha256(Path(g1).read_bytes()).hexdigest(),
 "qualification":{"rule":"locked G1 call-8 endpoint-hold accuracy >= .95", "status":selection, "call8_endpoint_hold_accuracy":float(call8)},
 "controller":{"form":"h(D+AB)+b", "rank":48, "replicas":2, "objective":"pure on-policy successor CE calls 1..16", "placement":"before frozen executor"},
 "analysis_source":source, "analysis_source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
 "log":log, "physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
}

if [[ "$selection_status" != qualified ]]; then
  write_manifest endpoint_failed
  echo "ENDPOINT_FAILED seed=$seed call8_accuracy=$call8_accuracy"
  exit 0
fi

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
echo "START $(date --iso-8601=seconds) SEED=$seed CALL8=$call8_accuracy"

for replica in 1 2; do
  controller_seed=$((2026095000 + 10 * seed + replica))
  validation_seed=$((2026097000 + 10 * seed + replica))
  out_dir="$controller_root/rank48_seed${replica}"
  if [[ -s "$out_dir/best_controller.pt" && -s "$out_dir/final_controller.pt" && -s "$out_dir/summary.json" ]]; then
    echo "CONTROLLER rank48_seed${replica} ALREADY_COMPLETE"; continue
  fi
  [[ ! -e "$out_dir" ]] || { echo "refusing partial/ambiguous controller directory: $out_dir" >&2; exit 2; }
  "$python_bin" -u "$trainer" train \
    --checkpoint "$checkpoint" --out-dir "$out_dir" --rank 48 --seed "$controller_seed" \
    --updates 8000 --max-train-call 16 --batch-size 128 --learning-rate 0.0001 --grad-clip 1.0 \
    --validation-every 400 --validation-seed "$validation_seed" --validation-batches 8 --validation-batch-size 256 --device cuda
done
echo "COMPLETE $(date --iso-8601=seconds)"
