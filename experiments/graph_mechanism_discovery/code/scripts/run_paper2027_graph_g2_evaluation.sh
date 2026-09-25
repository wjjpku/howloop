#!/usr/bin/env bash
# One G2 matched-interface evaluation on the fresh G1 locked test.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/wujiaju/paper2027_confirmatory/graph_g1_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
evaluator="$code_root/scripts/evaluate_paper2027_graph_g2.py"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
g1_summary="$run_root/evaluation/seed${seed}/summary.json"
locked_test="$run_root/locked/graph_permutations_512_all_starts.pt"
out_dir="$run_root/g2_interface/seed${seed}"
log_dir="/data/wujiaju/logs/paper2027_confirmatory/graph_g1_v1"
log_file="$log_dir/g2_interface_seed${seed}.log"
manifest="$run_root/manifests/g2_interface_seed${seed}.json"

if [[ -s "$out_dir/summary.json" ]]; then
  echo "G2 ALREADY_COMPLETE seed=$seed"; exit 0
fi
if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing partial/ambiguous G2 output directory: $out_dir" >&2; exit 2
fi
mkdir -p "$out_dir" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$g1_summary" && -s "$locked_test" && -s "$evaluator" ]] || { echo "missing G2 inputs" >&2; exit 2; }
qualification="$("$python_bin" - "$g1_summary" <<'PY'
import json, sys
summary=json.load(open(sys.argv[1])); row=next((r for r in summary["aggregate_rows"] if int(r["call"]) == 8),None)
if row is None: raise SystemExit("locked G1 summary has no call-8 row")
value=float(row["endpoint_hold_accuracy"]); print("qualified" if value >= .95 else "endpoint_failed", f"{value:.12g}")
PY
)"
read -r selection_status call8_accuracy <<< "$qualification"

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$g1_summary" "$locked_test" "$evaluator" "$log_file" "$selection_status" "$call8_accuracy" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
path,status,seed,checkpoint,g1,locked,source,log,selection,call8=sys.argv[1:]
payload={"status":status,"updated_at":datetime.now(timezone.utc).isoformat(),"backbone_seed":int(seed),"protocol_id":"paper2027.graph.g2.matched_interface.v1","checkpoint":checkpoint,"checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),"g1_locked_summary":g1,"g1_locked_summary_sha256":hashlib.sha256(Path(g1).read_bytes()).hexdigest(),"locked_test":locked,"locked_test_sha256":hashlib.sha256(Path(locked).read_bytes()).hexdigest(),"qualification":{"rule":"locked G1 call-8 endpoint-hold accuracy >= .95","status":selection,"call8_endpoint_hold_accuracy":float(call8)},"analysis_source":source,"analysis_source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"log":log,"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
}
if [[ "$selection_status" != qualified ]]; then
  write_manifest endpoint_failed; echo "ENDPOINT_FAILED seed=$seed call8_accuracy=$call8_accuracy"; exit 0
fi

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -u "$evaluator" --checkpoint "$checkpoint" --locked-test "$locked_test" --out-dir "$out_dir" \
  --source-calls 16 32 64 --permutations 512 --test-seed 2026093001 --random-seed "$((2026094001 + seed))" --batch-size 256 --device cuda
