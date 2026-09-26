#!/usr/bin/env bash
# Evaluate raw and all controller evidence exactly once on the final lock.
set -euo pipefail
run_root="${PAPER2027_GRAPH_G4_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g4_disjoint_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
evaluation_dir="${PAPER2027_GRAPH_G4_EVALUATION_DIR:-g4_evaluation}"
manifest_prefix="${PAPER2027_GRAPH_G4_EVALUATION_MANIFEST_PREFIX:-evaluation_seed}"
[[ "$evaluation_dir" != */* && "$evaluation_dir" != .* && "$evaluation_dir" != "" ]] || {
  echo "evaluation directory must be a simple child name" >&2; exit 2;
}
[[ "$manifest_prefix" != */* && "$manifest_prefix" != .* && "$manifest_prefix" != "" ]] || {
  echo "manifest prefix must be a simple filename prefix" >&2; exit 2;
}
code_root="$run_root/analysis_code"
runner_source="${BASH_SOURCE[0]}"
raw_evaluator="$code_root/scripts/evaluate_paper2027_graph_g4.py"
controller_evaluator="$code_root/reasoning_loop/paper2027_graph_g3_controller.py"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
final_lock="$run_root/locks/final_test_permutations_512.pt"
controller_root="$run_root/g4_controllers/seed${seed}"
out_root="$run_root/$evaluation_dir/seed${seed}"
log_dir="$run_root/logs"
log_file="$log_dir/${manifest_prefix}${seed}.log"
manifest="$run_root/manifests/${manifest_prefix}${seed}.json"
[[ -s "$raw_evaluator" && -s "$controller_evaluator" && -s "$checkpoint" && -s "$final_lock" ]] || { echo "G4 evaluation input missing" >&2; exit 2; }
[[ -s "$controller_root/rank48_seed1/best_controller.pt" && -s "$controller_root/rank48_seed2/best_controller.pt" ]] || { echo "G4 controller replicas missing" >&2; exit 2; }
if [[ -e "$out_root" ]]; then echo "refusing to overwrite G4 evaluation $out_root" >&2; exit 2; fi
mkdir -p "$out_root" "$log_dir" "$(dirname "$manifest")"
write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$final_lock" "$raw_evaluator" "$controller_evaluator" "$runner_source" "$evaluation_dir" "$manifest_prefix" "$log_file" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
path,status,seed,checkpoint,lock,raw,controller,runner,evaluation_dir,manifest_prefix,log=sys.argv[1:]
d=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p={"status":status,"updated_at":datetime.now(timezone.utc).isoformat(),"protocol_id":"paper2027.graph.g4.final_lock_evaluation.v1","backbone_seed":int(seed),"checkpoint":checkpoint,"checkpoint_sha256":d(checkpoint),"final_test_lock":lock,"final_test_lock_sha256":d(lock),"raw_source":raw,"raw_source_sha256":d(raw),"controller_source":controller,"controller_source_sha256":d(controller),"runner_source":runner,"runner_source_sha256":d(runner),"evaluation_dir":evaluation_dir,"evaluation_manifest_prefix":manifest_prefix,"full_controller_replicas":["rank48_seed1","rank48_seed2"],"appendix_modes":["executor_off","batch_shuffle","D_only","no_AB","identity_D","AB_only","no_bias","mean_D","shuffle_D","spectrum_matched_random_delta"],"log":log,"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
Path(path).write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
}
write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
echo "START $(date --iso-8601=seconds) SEED=$seed"
"$python_bin" -u "$raw_evaluator" --checkpoint "$checkpoint" --final-test-lock "$final_lock" --out-dir "$out_root/raw" --max-call 128 --batch-size 256 --device cuda
run_mode() {
  # Bash expands the entire ``local`` declaration before assigning its first
  # variable.  With ``set -u``, constructing ``controller`` from ``label`` in
  # the same declaration therefore fails.  Bind the arguments sequentially.
  local label="$1"
  local mode="$2"
  local controller="$controller_root/$label/best_controller.pt"
  "$python_bin" -u "$controller_evaluator" evaluate --checkpoint "$checkpoint" --controller "$controller" --locked-test "$final_lock" --out-dir "$out_root/$label/$mode" --mode "$mode" --permutations 512 --test-seed 2026081202 --random-seed "$((2026098000 + seed))" --max-call 128 --batch-size 256 --require-final-test-lock-excluded --protocol-id paper2027.graph.g4.final_lock_controller_evaluation.v1 --device cuda
}
run_mode rank48_seed1 full
run_mode rank48_seed2 full
for mode in executor_off batch_shuffle D_only no_AB identity_D AB_only no_bias mean_D shuffle_D spectrum_matched_random_delta; do run_mode rank48_seed1 "$mode"; done
echo "COMPLETE $(date --iso-8601=seconds)"
