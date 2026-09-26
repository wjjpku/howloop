#!/usr/bin/env bash
# One graph-disjoint N8 D8L8 final-only backbone.  The two locks must already
# exist; both are rejected by the optimiser's only graph sampler.
set -euo pipefail

run_root="${PAPER2027_GRAPH_G4_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g4_disjoint_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/code"
trainer="$code_root/reasoning_loop/paper2027_graph_g4_backbone.py"
selection_lock="$run_root/locks/selection_permutations_512.pt"
final_lock="$run_root/locks/final_test_permutations_512.pt"
out_dir="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}"
log_dir="$run_root/logs"
log_file="$log_dir/backbone_seed${seed}.log"
manifest="$run_root/manifests/backbone_seed${seed}.json"

mkdir -p "$(dirname "$out_dir")" "$log_dir" "$(dirname "$manifest")"
[[ -s "$trainer" && -s "$selection_lock" && -s "$final_lock" ]] || { echo "G4 source or locks missing" >&2; exit 2; }
if [[ -e "$out_dir" ]]; then echo "refusing to overwrite G4 backbone $out_dir" >&2; exit 2; fi

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$trainer" "$selection_lock" "$final_lock" "$out_dir" "$log_file" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path,status,seed,source,selection,final,out_dir,log=sys.argv[1:]
digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
payload={"status":status,"updated_at":datetime.now(timezone.utc).isoformat(),"protocol_id":"paper2027.graph.g4.disjoint_backbone.v1","backbone_seed":int(seed),"architecture":{"node_count":8,"max_depth":8,"d_model":256,"n_heads":4,"d_mlp":1024,"physical_blocks":2,"trained_calls":8,"block_schedule":"all_blocks","normalization":"pre_layernorm"},"training":{"loss":"final-only successor CE at call 8","updates":20000,"batch_size":512,"weight_decay":0.3,"learning_rate":0.0003,"warmup_steps":500,"trajectory_aux_weight":0.0,"aux_loss":0.0,"sampler":"reject both immutable held-out locks"},"source":source,"source_sha256":digest(source),"selection_lock":selection,"selection_lock_sha256":digest(selection),"final_test_lock":final,"final_test_lock_sha256":digest(final),"output_dir":out_dir,"log":log,"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
}
write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
echo "START $(date --iso-8601=seconds) SEED=$seed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$python_bin" -u "$trainer" --out-dir "$out_dir" --selection-lock "$selection_lock" --final-test-lock "$final_lock" --seed "$seed" --steps 20000 --batch-size 512 --eval-batch-size 512 --eval-every 1000 --learning-rate 0.0003 --weight-decay 0.3 --warmup-steps 500 --grad-clip 1.0 --amp --device cuda
echo "COMPLETE $(date --iso-8601=seconds)"
