#!/usr/bin/env bash
# Locked-test G3 evaluation.  Full controller replicas are primary evidence;
# component and random-orientation views of replica 1 are appendix evidence.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g1_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
evaluator="$code_root/reasoning_loop/paper2027_graph_g3_controller.py"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
controller_root="$run_root/g3_controllers/seed${seed}"
locked_test="$run_root/locked/graph_permutations_512_all_starts.pt"
out_root="$run_root/g3_evaluation/seed${seed}"
log_dir="/data/paperexperiment/logs/paper2027_confirmatory/graph_g1_v1"
log_file="$log_dir/g3_evaluation_seed${seed}.log"
manifest="$run_root/manifests/g3_evaluation_seed${seed}.json"

if [[ -s "$out_root/rank48_seed1/full/summary.json" && -s "$out_root/rank48_seed2/full/summary.json" && -s "$out_root/rank48_seed1/executor_off/summary.json" ]]; then
  echo "G3 EVALUATION ALREADY_COMPLETE seed=$seed"; exit 0
fi
if [[ -d "$out_root" ]] && [[ -n "$(find "$out_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing partial/ambiguous G3 evaluation directory: $out_root" >&2; exit 2
fi
mkdir -p "$out_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$locked_test" && -s "$evaluator" ]] || { echo "missing G3 evaluation inputs" >&2; exit 2; }
[[ -s "$controller_root/rank48_seed1/best_controller.pt" && -s "$controller_root/rank48_seed2/best_controller.pt" ]] || { echo "missing selected G3 controller replicas" >&2; exit 2; }

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$locked_test" "$evaluator" "$log_file" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, seed, checkpoint, locked, source, log = sys.argv[1:]
payload={"status":status,"updated_at":datetime.now(timezone.utc).isoformat(),"backbone_seed":int(seed),"protocol_id":"paper2027.graph.g3.locked_evaluation.v1","checkpoint":checkpoint,"checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),"locked_test":locked,"locked_test_sha256":hashlib.sha256(Path(locked).read_bytes()).hexdigest(),"analysis_source":source,"analysis_source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"full_controller_replicas":["rank48_seed1","rank48_seed2"],"appendix_modes":["raw","D_only","no_AB","identity_D","AB_only","no_bias","mean_D","shuffle_D","batch_shuffle","spectrum_matched_random_delta","executor_off"],"log":log,"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY
}

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
echo "START $(date --iso-8601=seconds) SEED=$seed"

run_mode() {
  local label="$1" mode="$2" controller="$controller_root/$label/best_controller.pt"
  "$python_bin" -u "$evaluator" evaluate --checkpoint "$checkpoint" --controller "$controller" --locked-test "$locked_test" \
    --out-dir "$out_root/$label/$mode" --mode "$mode" --permutations 512 --test-seed 2026093001 \
    --random-seed "$((2026098000 + seed))" --max-call 128 --batch-size 256 --device cuda
}
run_mode rank48_seed1 full
run_mode rank48_seed2 full
for mode in raw D_only no_AB identity_D AB_only no_bias mean_D shuffle_D batch_shuffle spectrum_matched_random_delta executor_off; do
  run_mode rank48_seed1 "$mode"
done
echo "COMPLETE $(date --iso-8601=seconds)"
