#!/usr/bin/env bash
# One-shot, manifest-gated post-training pipeline for corrected input-once
# Parity.  This deliberately has no tmux/self-session waiting logic: launch
# it only after the backbone cohort has ended, and it fails closed otherwise.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:?set PAPER2027_PARITY_ROOT}"
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="${PAPER2027_PARITY_CODE_ROOT:-$run_root/analysis_code}"
scripts_dir="$code_root/scripts"
log_dir="$run_root/logs"
log_file="$log_dir/parity_posttraining_pipeline.log"
manifest="$run_root/manifests/posttraining_pipeline.json"
pipeline_source="${BASH_SOURCE[0]}"
seeds="${PAPER2027_PARITY_ENDPOINT_SEEDS:-3 4 5 6 7 8 9 10 11 12 13 14}"
deep_seeds="${PAPER2027_PARITY_DEEP_SEEDS:-3 4 5}"

[[ ! -e "$manifest" ]] || { echo "refusing existing post-training manifest: $manifest" >&2; exit 2; }
for script in \
  run_paper2027_parity_p1_evaluation_campaign.sh \
  run_paper2027_parity_p2_campaign.sh \
  run_paper2027_parity_p4_controller_mechanism.sh \
  aggregate_paper2027_parity_p1.py \
  aggregate_paper2027_parity_p2.py \
  make_paper2027_parity_p1_figures.py \
  make_paper2027_parity_p2_figures.py; do
  [[ -s "$scripts_dir/$script" ]] || { echo "missing pipeline input: $scripts_dir/$script" >&2; exit 2; }
done

for seed in $seeds; do
  manifest_path="$run_root/manifests/backbone_seed${seed}.json"
  "$python_bin" - "$manifest_path" "$seed" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
    raise SystemExit(f"P1 backbone seed {sys.argv[2]} is not complete")
PY
done

mkdir -p "$log_dir" "$(dirname "$manifest")"
"$python_bin" - "$manifest" "$run_root" "$code_root" "$seeds" "$deep_seeds" "$pipeline_source" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, root, code, seeds, deep, pipeline = sys.argv[1:]
sources = {}
for name in (
    "run_paper2027_parity_p1_evaluation_campaign.sh",
    "run_paper2027_parity_p2_campaign.sh",
    "run_paper2027_parity_p4_controller_mechanism.sh",
    "aggregate_paper2027_parity_p1.py",
    "aggregate_paper2027_parity_p2.py",
    "make_paper2027_parity_p1_figures.py",
    "make_paper2027_parity_p2_figures.py",
):
    source = Path(code) / "scripts" / name
    sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
Path(path).write_text(json.dumps({
    "status": "running", "updated_at": datetime.now(timezone.utc).isoformat(),
    "protocol_id": "paper2027.parity.posttraining_pipeline.v1",
    "run_root": root, "analysis_code_root": code,
    "pipeline_source": pipeline, "pipeline_source_sha256": hashlib.sha256(Path(pipeline).read_bytes()).hexdigest(),
    "backbone_seeds": [int(x) for x in seeds.split()],
    "deep_seeds": [int(x) for x in deep.split()], "source_sha256": sources,
}, indent=2, sort_keys=True) + "\n")
PY

finish() {
  local exit_status=$?
  "$python_bin" - "$manifest" "$exit_status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
payload = json.load(open(path))
payload["status"] = "complete" if int(status) == 0 else "failed"
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
open(path, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  exit "$exit_status"
}
trap finish EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

echo "P1 evaluation start $(date --iso-8601=seconds)"
PAPER2027_PARITY_ROOT="$run_root" PAPER2027_PARITY_ENDPOINT_SEEDS="$seeds" \
  PAPER2027_PARITY_DEEP_SEEDS="$deep_seeds" PAPER2027_PYTHON="$python_bin" \
  bash "$scripts_dir/run_paper2027_parity_p1_evaluation_campaign.sh"

echo "P2 prospective gate start $(date --iso-8601=seconds)"
PAPER2027_PARITY_ROOT="$run_root" PAPER2027_PARITY_P2_SEEDS="$deep_seeds" \
  PAPER2027_PYTHON="$python_bin" bash "$scripts_dir/run_paper2027_parity_p2_campaign.sh"

# P4 is a mechanism appendix for seed 5 only.  It exits successfully without
# artifacts if the prospectively selected boundary has no disease.
echo "P4 mechanism gate start $(date --iso-8601=seconds)"
PAPER2027_PARITY_ROOT="$run_root" PAPER2027_PARITY_SEED=5 PAPER2027_PYTHON="$python_bin" \
  bash "$scripts_dir/run_paper2027_parity_p4_controller_mechanism.sh"

aggregate_root="$run_root/aggregate"
echo "aggregate start $(date --iso-8601=seconds)"
"$python_bin" "$scripts_dir/aggregate_paper2027_parity_p1.py" --root "$run_root" \
  --out-dir "$aggregate_root/p1" --seeds $seeds --deep-seeds $deep_seeds --bootstrap-draws 10000
"$python_bin" "$scripts_dir/aggregate_paper2027_parity_p2.py" --root "$run_root" \
  --out-dir "$aggregate_root/p2" --seeds $deep_seeds
"$python_bin" "$scripts_dir/make_paper2027_parity_p1_figures.py" \
  --aggregate-dir "$aggregate_root/p1" --figure-dir "$run_root/figures" \
  --results-tex "$aggregate_root/p1/results.tex"
"$python_bin" "$scripts_dir/make_paper2027_parity_p2_figures.py" \
  --aggregate-dir "$aggregate_root/p2" --figure-dir "$run_root/figures"
echo "complete $(date --iso-8601=seconds)"
