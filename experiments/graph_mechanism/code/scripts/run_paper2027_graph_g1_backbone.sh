#!/usr/bin/env bash
# One fresh final-only N8 D8L8 Graph backbone. The seed is the top-level
# replication unit; GPU placement is logged but never treated as a seed.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/wujiaju/paper2027_confirmatory/graph_g1_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_file="$run_root/code/graph_path_loop.py"
out_base="$run_root/backbones/seed${seed}"
out_dir="$out_base/graphpath_N8_D8_d256_B2_L8_seed${seed}"
log_dir="/data/wujiaju/logs/paper2027_confirmatory/graph_g1_v1"
log_file="$log_dir/backbone_seed${seed}.log"
manifest="$run_root/manifests/backbone_seed${seed}.json"

mkdir -p "$out_base" "$log_dir" "$(dirname "$manifest")"
if [[ -e "$out_dir/best.pt" || -e "$out_dir/final.pt" ]]; then
  echo "refusing to overwrite a Graph G1 backbone: $out_dir" >&2; exit 2
fi
[[ -s "$code_file" ]] || { echo "missing frozen Graph training source" >&2; exit 2; }

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$code_file" "$out_dir" "$log_file" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, seed, code, out_dir, log = sys.argv[1:]
payload = {
  "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
  "backbone_seed": int(seed), "protocol_id": "paper2027.graph.g1.phenotype.v1",
  "architecture": {"node_count":8,"max_depth":8,"d_model":256,"n_heads":4,"d_mlp":1024,"physical_blocks":2,"trained_calls":8,"block_schedule":"all_blocks","normalization":"pre_layernorm"},
  "training": {"loss":"final-only successor CE at call 8","updates":20000,"batch_size":512,"weight_decay":0.3,"learning_rate":0.0003,"warmup_steps":500,"trajectory_aux_weight":0.0,"aux_loss":0.0},
  "source": code, "source_sha256": hashlib.sha256(Path(code).read_bytes()).hexdigest(),
  "output_dir": out_dir, "log": log,
  "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
echo "START $(date --iso-8601=seconds) SEED=$seed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$python_bin" -u "$code_file" \
  --node-count 8 --max-depth 8 --d-model 256 --n-heads 4 --d-mlp 1024 --n-layers 2 \
  --loops 8 --steps 20000 --batch-size 512 --eval-batch-size 1024 --eval-batches 16 \
  --eval-every 1000 --print-every 1000 --lr 0.0003 --weight-decay 0.3 \
  --warmup-steps 500 --grad-clip 1.0 --seed "$seed" --dropout 0.0 --aux-loss 0.0 \
  --trajectory-aux-weight 0.0 --device cuda --amp --no-compile --save-checkpoints \
  --out-dir "$out_base"
echo "COMPLETE $(date --iso-8601=seconds)"
