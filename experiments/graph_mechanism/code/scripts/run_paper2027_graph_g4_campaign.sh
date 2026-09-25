#!/usr/bin/env bash
# Manifest-gated G4 continuation: do not inspect or select on final-lock data
# before every registered backbone has finished.  This scheduler is not an
# experimental source; all scientific artifacts are still created by the
# per-backbone/controller/evaluation runners and carry their own hashes.
set -euo pipefail

run_root="${PAPER2027_GRAPH_G4_ROOT:-/data/wujiaju/paper2027_confirmatory/graph_g4_disjoint_v3}"
python_bin="${PAPER2027_PYTHON:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
code_root="$run_root/code"
analysis_root="$run_root/analysis_code"
seeds="100 101 102 103 104 105 106 107 108 109 110 111"
log="$run_root/logs/campaign.log"
mkdir -p "$run_root/logs"
exec >> "$log" 2>&1

manifest_status() {
  "$python_bin" - "$run_root" "$@" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); kind = sys.argv[2]; seeds = [int(x) for x in sys.argv[3:]]
prefix = {"backbone": "backbone_seed", "controller": "controller_seed", "evaluation": "evaluation_seed"}[kind]
states = {}
for seed in seeds:
    path = root / "manifests" / f"{prefix}{seed}.json"
    states[seed] = json.loads(path.read_text()).get("status") if path.is_file() else "missing"
print(" ".join(f"{seed}:{states[seed]}" for seed in seeds))
if any(value == "failed" for value in states.values()): raise SystemExit(2)
raise SystemExit(0 if all(value == "complete" for value in states.values()) else 1)
PY
}

wait_complete() {
  local kind="$1"
  while true; do
    if status="$(manifest_status "$kind" $seeds)"; then
      echo "$(date --iso-8601=seconds) $kind complete: $status"; return
    fi
    exit_status=$?
    if [[ "$exit_status" -eq 2 ]]; then
      echo "$(date --iso-8601=seconds) $kind manifest failed: $status" >&2; exit 2
    fi
    echo "$(date --iso-8601=seconds) waiting $kind: $status"
    sleep 60
  done
}

echo "$(date --iso-8601=seconds) campaign waiting for all G4 backbones"
wait_complete backbone
for pair in "3:100,105,110" "4:101,106,111" "5:102,107" "6:103,108" "7:104,109"; do
  gpu="${pair%%:*}"; seed_csv="${pair#*:}"; seed_list="$(printf '%s' "$seed_csv" | tr ',' ' ')"
  session="paper2027_graph_g4v3_controller_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "controller session already exists: $session" >&2; exit 2
  fi
  command="cd $code_root && for seed in $seed_list; do PAPER2027_GRAPH_SEED=\$seed CUDA_VISIBLE_DEVICES=$gpu PAPER2027_GRAPH_G4_ROOT=$run_root PAPER2027_PYTHON=$python_bin bash scripts/run_paper2027_graph_g4_controller.sh; PAPER2027_GRAPH_SEED=\$seed CUDA_VISIBLE_DEVICES=$gpu PAPER2027_GRAPH_G4_ROOT=$run_root PAPER2027_PYTHON=$python_bin bash scripts/run_paper2027_graph_g4_evaluation.sh; done"
  tmux new-session -d -s "$session" "$command"
  echo "$(date --iso-8601=seconds) launched $session seeds=$seed_csv"
done
wait_complete controller
wait_complete evaluation
PYTHONPATH="$analysis_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" "$analysis_root/scripts/aggregate_paper2027_graph_g4.py" --root "$run_root" --out-dir "$run_root/aggregate" --bootstrap-draws 10000
PYTHONPATH="$analysis_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" "$analysis_root/scripts/make_paper2027_graph_g4_figures.py" --root "$run_root" --out-dir "$run_root/figures" --max-call 64
echo "$(date --iso-8601=seconds) G4 campaign complete"
