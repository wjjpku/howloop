#!/usr/bin/env bash
# Evaluate a completed P1 backbone against a frozen analysis-code snapshot.
# Every command passes --paper-mode, so an accidentally legacy every-loop
# checkpoint cannot enter the corrected Parity result table.
set -euo pipefail

run_root="${PAPER2027_PARITY_ROOT:-/data/paperexperiment/paper2027_confirmatory/parity_input_once_v2}"
seed="${PAPER2027_PARITY_SEED:?set PAPER2027_PARITY_SEED}"
mode="${PAPER2027_PARITY_EVALUATION_MODE:-endpoint}" # endpoint | deep
python_bin="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
code_root="$run_root/analysis_code"
checkpoint="$run_root/backbones/parity_input_once_seed${seed}/final.pt"
out_root="$run_root/evaluation/seed${seed}"
log_dir="/data/paperexperiment/logs/paper2027_confirmatory/parity_input_once_v2"
log_file="$log_dir/evaluation_seed${seed}_${mode}.log"
manifest="$run_root/manifests/evaluation_seed${seed}_${mode}.json"

case "$mode" in endpoint|deep) ;; *) echo "mode must be endpoint or deep" >&2; exit 2 ;; esac
if [[ -s "$manifest" ]]; then
  if grep -q '"status": "complete"' "$manifest"; then
    echo "P1 EVALUATION ALREADY_COMPLETE seed=$seed mode=$mode"; exit 0
  fi
  echo "refusing partial/ambiguous P1 evaluation manifest: $manifest" >&2; exit 2
fi
if [[ -d "$out_root" ]] && [[ -n "$(find "$out_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing partial/ambiguous P1 evaluation output directory: $out_root" >&2; exit 2
fi
mkdir -p "$out_root" "$log_dir" "$(dirname "$manifest")"
[[ -s "$checkpoint" ]] || { echo "missing completed P1 checkpoint: $checkpoint" >&2; exit 2; }
[[ -s "$code_root/reasoning_loop/paper_length_telomere.py" ]] || {
  echo "missing frozen analysis code under $code_root" >&2; exit 2;
}

write_manifest() {
  local status="$1"
  "$python_bin" - "$manifest" "$status" "$seed" "$mode" "$checkpoint" "$code_root" "$out_root" "$log_file" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

path, status, seed, mode, checkpoint, code_root, out_root, log_file = sys.argv[1:]
root = Path(code_root)
sources = {}
for relative in (
    "reasoning_loop/paper_length_telomere.py",
    "reasoning_loop/analyze_parity_four_phase.py",
    "scripts/evaluate_parity_far_horizon.py",
    "scripts/evaluate_parity_loop_depth_heatmap.py",
    "scripts/select_parity_prospective_boundary.py",
):
    source = root / relative
    if source.exists():
        sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
payload = {
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getpid(),
    "backbone_seed": int(seed),
    "evaluation_mode": mode,
    "protocol_id": "paper2027.parity.p1.input_once.v2",
    "checkpoint": checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    "paper_mode": True,
    "analysis_code_root": code_root,
    "analysis_source_sha256": sources,
    "log": log_file,
    "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
outputs = {}
for relative in (
    "endpoint_horizon/horizon.csv",
    "endpoint_horizon/summary.json",
    "diagnose/trajectory.csv",
    "diagnose/summary.json",
    "four_phase/summary.json",
    "loop_depth_heatmap/loop_depth_metrics.csv",
    "loop_depth_heatmap/summary.json",
    "prospective_boundary.json",
):
    artifact = Path(out_root) / relative
    if artifact.exists():
        outputs[relative] = hashlib.sha256(artifact.read_bytes()).hexdigest()
payload["output_sha256"] = outputs
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_manifest running
trap 'status=$?; if [[ $status -eq 0 ]]; then write_manifest complete; else write_manifest failed; fi; exit $status' EXIT
exec >> "$log_file" 2>&1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

echo "START $(date --iso-8601=seconds)"
echo "SEED=$seed MODE=$mode CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "STAGE endpoint_horizon START $(date --iso-8601=seconds)"
"$python_bin" -u "$code_root/scripts/evaluate_parity_far_horizon.py" \
  --task parity --checkpoint "$checkpoint" --paper-mode \
  --lengths 10 20 24 32 40 48 64 80 100 128 160 200 250 300 400 500 \
  --examples 512 --max-batch-size 32 --token-budget 8192 \
  --seed "$((2026082701 + seed))" --device cuda \
  --out-dir "$out_root/endpoint_horizon"
echo "STAGE endpoint_horizon COMPLETE $(date --iso-8601=seconds)"

if [[ "$mode" == "deep" ]]; then
  echo "STAGE diagnose START $(date --iso-8601=seconds)"
  "$python_bin" -u "$code_root/reasoning_loop/paper_length_telomere.py" diagnose \
    --checkpoint "$checkpoint" --paper-mode \
    --lengths 10 20 24 32 40 48 64 80 100 \
    --batch-size 128 --batches 8 --maximum-step 112 \
    --seed "$((2026082801 + seed))" --device cuda \
    --out-dir "$out_root/diagnose"
  echo "STAGE diagnose COMPLETE $(date --iso-8601=seconds)"

  echo "STAGE four_phase START $(date --iso-8601=seconds)"
  "$python_bin" -u "$code_root/reasoning_loop/analyze_parity_four_phase.py" \
    --checkpoint "$checkpoint" --paper-mode --out-dir "$out_root/four_phase" \
    --device cuda --batch-size 256 --batches 2 --causal-batch-size 256 \
    --discovery-lengths 12 16 20 24 32 40 64 80 \
    --evaluation-lengths 10 14 18 22 28 36 48 72 100 \
    --causal-lengths 10 22 36 72 \
    --random-controls 100 \
    --discovery-seed "$((2026082901 + seed))" \
    --evaluation-seed "$((2026083001 + seed))" \
    --causal-seed "$((2026083101 + seed))"
  echo "STAGE four_phase COMPLETE $(date --iso-8601=seconds)"

  echo "STAGE heatmap START $(date --iso-8601=seconds)"
  "$python_bin" -u "$code_root/scripts/evaluate_parity_loop_depth_heatmap.py" \
    --checkpoint "$checkpoint" --paper-mode \
    --out-dir "$out_root/loop_depth_heatmap" \
    --min-length 1 --max-length 100 --max-loop 112 \
    --examples 256 --batch-size 64 --seed "$((2026083201 + seed))" --device cuda
  echo "STAGE heatmap COMPLETE $(date --iso-8601=seconds)"

  echo "STAGE prospective_boundary START $(date --iso-8601=seconds)"
  "$python_bin" -u "$code_root/scripts/select_parity_prospective_boundary.py" \
    --heatmap-csv "$out_root/loop_depth_heatmap/loop_depth_metrics.csv" \
    --minimum-length 21 --maximum-length 100 --band-cap 200 \
    --out "$out_root/prospective_boundary.json"
  echo "STAGE prospective_boundary COMPLETE $(date --iso-8601=seconds)"
fi

echo "COMPLETE $(date --iso-8601=seconds)"
