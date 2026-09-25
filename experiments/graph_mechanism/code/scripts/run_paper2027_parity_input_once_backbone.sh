#!/usr/bin/env bash
# One independently initialized backbone for the preregistered Parity P1 run.
# Run exactly one copy per physical GPU; the launcher records the physical GPU
# in its manifest, while the result itself remains the backbone seed.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/wujiaju/paper2027_confirmatory/parity_input_once_v2}"
seed="${PAPER2027_PARITY_SEED:?set PAPER2027_PARITY_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_file="$run_root/code/paper_length_telomere.py"
out_dir="$run_root/backbones/parity_input_once_seed${seed}"
log_dir="/data/wujiaju/logs/paper2027_confirmatory/parity_input_once_v2"
log_file="$log_dir/backbone_seed${seed}.log"
pid_file="$run_root/manifests/backbone_seed${seed}.pid"
manifest="$run_root/manifests/backbone_seed${seed}.json"

mkdir -p "$out_dir" "$log_dir" "$(dirname "$pid_file")"
if [[ -e "$out_dir/best.pt" || -e "$out_dir/final.pt" ]]; then
  echo "refusing to overwrite an existing backbone artifact: $out_dir" >&2
  exit 2
fi
if [[ ! -s "$code_file" ]]; then
  echo "missing frozen experiment source: $code_file" >&2
  exit 2
fi

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$code_file" "$out_dir" "$log_file" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, seed, code, out_dir, log_file = sys.argv[1:]
code_path = Path(code)
payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "launcher_pid": os.getpid(),
    "backbone_seed": int(seed),
    "experiment": "P1 input-once Parity backbone replication",
    "protocol_id": "paper2027.parity.p1.input_once.v2",
    "architecture": {
        "attention": "causal NoPE",
        "shared_physical_layers": 1,
        "d_model": 256,
        "attention_heads": 64,
        "mlp_width": 1024,
        "token_embedding_injection": "initial_only",
        "position_embedding": "none",
        "final_layer_norm": "after every recurrent call",
    },
    "training": {
        "logical_lengths": [1, 20],
        "loss": "answer-region CE only at T(n)=n",
        "updates": 100001,
        "batch_size": 64,
        "precision": "fp32",
        "supervision": "adaptive_step",
    },
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "source": str(code_path),
    "source_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
    "output_dir": out_dir,
    "log": log_file,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

echo $$ > "$pid_file"
write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1

echo "START $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUTPUT=$out_dir"
echo "SOURCE_SHA256=$(sha256sum "$code_file" | awk '{print $1}')"

"$python_bin" -u "$code_file" backbone \
  --task parity \
  --supervision adaptive_step \
  --official-model-config \
  --paper-mode \
  --token-embedding-injection initial_only \
  --position-embedding none \
  --position-injection initial_only \
  --steps 100001 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --seed "$seed" \
  --device auto \
  --no-amp \
  --log-every 100 \
  --eval-every 1000 \
  --eval-batch-size 256 \
  --eval-batches 4 \
  --checkpoint-every 10000 \
  --out-dir "$out_dir"

echo "COMPLETE $(date --iso-8601=seconds)"
