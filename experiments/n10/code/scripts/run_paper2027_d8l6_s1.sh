#!/usr/bin/env bash
# Appendix-only matched D8L6 one-vs-two stage selection.  Both objectives use
# exactly the same pure-CE controller protocol; only the post-F target differs.
set -euo pipefail

root="${PAPER2027_D8L6_ROOT:-/data/paperexperiment/paper2027_confirmatory/d8l6_s1_v1}"
code="$root/analysis_code"
py="${PAPER2027_PYTHON:-/data/paperexperiment/.venvs/loopreasoner/bin/python}"
checkpoint="${PAPER2027_D8L6_CHECKPOINT:-/data/paperexperiment/graph_path_functional_multiseed_20260725/training/D8_L6_seed6/graphpath_N8_D8_d256_B2_L6_seed6/best.pt}"
trainer="$code/reasoning_loop/paper2027_d8l6_s1.py"
locked="$root/locked/graph_permutations_512_all_starts.pt"
log_dir="/data/paperexperiment/logs/paper2027_confirmatory/d8l6_s1_v1"
manifest="$root/manifest.json"

[[ -s "$checkpoint" && -s "$trainer" ]] || { echo "missing D8L6 checkpoint or frozen S1 code" >&2; exit 2; }
mkdir -p "$root/controllers" "$root/evaluation" "$root/locked" "$log_dir"
if [[ -s "$manifest" ]] && grep -q '"status": "complete"' "$manifest"; then echo "D8L6 S1 ALREADY_COMPLETE"; exit 0; fi
if [[ -e "$manifest" ]]; then echo "refusing partial/ambiguous D8L6 S1 manifest" >&2; exit 2; fi

"$py" - "$manifest" "$checkpoint" "$trainer" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
path,checkpoint,trainer=sys.argv[1:]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
Path(path).write_text(json.dumps({"status":"running","updated_at":datetime.now(timezone.utc).isoformat(),"protocol_id":"paper2027.d8l6.s1.matched_pure_ce.v1","checkpoint":checkpoint,"checkpoint_sha256":h(checkpoint),"trainer":trainer,"trainer_sha256":h(trainer),"controller":{"form":"h(D+AB)+b","rank":48,"replicas_per_target":2,"updates":8000,"objective":"pure post-executor CE from same raw h6"}},indent=2,sort_keys=True)+"\n")
PY
trap 'status=$?; if [[ $status -eq 0 ]]; then sed -i.bak "s/\"running\"/\"complete\"/" "$manifest"; rm -f "$manifest.bak"; else sed -i.bak "s/\"running\"/\"failed\"/" "$manifest"; rm -f "$manifest.bak"; fi; exit $status' EXIT
export PYTHONPATH="$code${PYTHONPATH:+:$PYTHONPATH}"
exec >> "$log_dir/s1.log" 2>&1
for hop in 1 2; do
  for replica in 1 2; do
    out="$root/controllers/hop${hop}_seed${replica}"
    [[ ! -e "$out" ]] || { echo "refusing existing controller dir: $out" >&2; exit 2; }
    seed=$((2026100000 + 100*hop + replica))
    "$py" -u "$trainer" train --checkpoint "$checkpoint" --out-dir "$out" --target-hop "$hop" --seed "$seed" --rank 48 --updates 8000 --batch-size 128 --learning-rate 0.0001 --grad-clip 1.0 --validation-every 400 --validation-seed "$((seed+10000))" --validation-batches 8 --validation-batch-size 256 --device cuda
    "$py" -u "$trainer" evaluate --checkpoint "$checkpoint" --controller "$out/best_controller.pt" --locked-test "$locked" --out-dir "$root/evaluation/hop${hop}_seed${replica}" --permutations 512 --test-seed 2026099501 --batch-size 256 --device cuda
  done
done
