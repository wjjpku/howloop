#!/usr/bin/env bash
# Fit two controllers for every G4 backbone. No final-lock metric selects a
# seed; both locks are rejected during controller training.
set -euo pipefail
run_root="${PAPER2027_GRAPH_G4_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g4_disjoint_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
trainer="$code_root/reasoning_loop/paper2027_graph_g3_controller.py"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
selection_lock="$run_root/locks/selection_permutations_512.pt"
final_lock="$run_root/locks/final_test_permutations_512.pt"
controller_root="$run_root/g4_controllers/seed${seed}"
log_dir="$run_root/logs"
log_file="$log_dir/controller_seed${seed}.log"
manifest="$run_root/manifests/controller_seed${seed}.json"
mkdir -p "$controller_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$trainer" && -s "$checkpoint" && -s "$selection_lock" && -s "$final_lock" ]] || { echo "G4 controller input missing" >&2; exit 2; }
if [[ -s "$controller_root/rank48_seed1/best_controller.pt" && -s "$controller_root/rank48_seed2/best_controller.pt" ]]; then echo "G4 controllers already complete seed=$seed"; exit 0; fi
if [[ -n "$(find "$controller_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then echo "refusing partial G4 controller directory" >&2; exit 2; fi
write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$trainer" "$selection_lock" "$final_lock" "$log_file" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
path,status,seed,checkpoint,source,selection,final,log=sys.argv[1:]
d=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p={"status":status,"updated_at":datetime.now(timezone.utc).isoformat(),"protocol_id":"paper2027.graph.g4.disjoint_controller.v1","backbone_seed":int(seed),"checkpoint":checkpoint,"checkpoint_sha256":d(checkpoint),"controller":{"form":"h(D+AB)+b","rank":48,"replicas":2,"objective":"pure on-policy successor CE calls 1..16","placement":"before frozen executor"},"selection_lock":selection,"selection_lock_sha256":d(selection),"final_test_lock":final,"final_test_lock_sha256":d(final),"analysis_source":source,"analysis_source_sha256":d(source),"log":log,"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
Path(path).write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
}
write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
echo "START $(date --iso-8601=seconds) SEED=$seed"
for replica in 1 2; do
  out_dir="$controller_root/rank48_seed${replica}"
  controller_seed=$((2026095000 + 10 * seed + replica))
  validation_seed=$((2026097000 + 10 * seed + replica))
  "$python_bin" -u "$trainer" train --checkpoint "$checkpoint" --out-dir "$out_dir" --rank 48 --seed "$controller_seed" --updates 8000 --max-train-call 16 --batch-size 128 --learning-rate 0.0001 --grad-clip 1.0 --validation-every 400 --validation-seed "$validation_seed" --validation-batches 8 --validation-batch-size 256 --train-exclude-locks "$selection_lock" "$final_lock" --validation-lock "$selection_lock" --protocol-id paper2027.graph.g4.disjoint_controller.v1 --device cuda
done
echo "COMPLETE $(date --iso-8601=seconds)"
