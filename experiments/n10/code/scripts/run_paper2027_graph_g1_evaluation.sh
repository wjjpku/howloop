#!/usr/bin/env bash
# Apply the same locked graph-permutation test to one final G1 checkpoint.
set -euo pipefail

run_root="${PAPER2027_GRAPH_ROOT:-/data/paperexperiment/paper2027_confirmatory/graph_g1_v1}"
seed="${PAPER2027_GRAPH_SEED:?set PAPER2027_GRAPH_SEED}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
checkpoint="$run_root/backbones/seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
locked_test="$run_root/locked/graph_permutations_512_all_starts.pt"
out_dir="$run_root/evaluation/seed${seed}"
log="/data/paperexperiment/logs/paper2027_confirmatory/graph_g1_v1/evaluation_seed${seed}.log"
manifest="$run_root/manifests/evaluation_seed${seed}.json"

if [[ -s "$out_dir/summary.json" && -s "$manifest" ]] && grep -q '"status": "complete"' "$manifest"; then
  echo "G1 EVALUATION ALREADY_COMPLETE seed=$seed"; exit 0
fi
if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing partial/ambiguous G1 evaluation directory: $out_dir" >&2; exit 2
fi
mkdir -p "$out_dir" "$(dirname "$log")" "$(dirname "$manifest")"
[[ -s "$checkpoint" && -s "$code_root/scripts/evaluate_paper2027_graph_g1.py" ]] || {
  echo "missing G1 checkpoint or analysis source" >&2; exit 2;
}

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$checkpoint" "$code_root" "$log" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, seed, checkpoint, root, log = sys.argv[1:]
analysis = Path(root) / "scripts/evaluate_paper2027_graph_g1.py"
payload = {
  "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
  "backbone_seed": int(seed), "protocol_id": "paper2027.graph.g1.phenotype.v1",
  "checkpoint": checkpoint,
  "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
  "analysis_source": str(analysis),
  "analysis_source_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
  "log": log, "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -u "$code_root/scripts/evaluate_paper2027_graph_g1.py" \
  --checkpoint "$checkpoint" --locked-test "$locked_test" --out-dir "$out_dir" \
  --permutations 512 --test-seed 2026093001 --max-call 128 --batch-size 256 --device cuda
