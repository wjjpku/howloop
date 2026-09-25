#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set one physical GPU}"

python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
repo_dir=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase_summary=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
source_bank=/data/wujiaju/graph_path_fixed_h1_j_20260801/wsd_identity_2x/r64_s16/focused5_h1/age_specific_j_bank_after_T05_R5.pt
out_dir=/data/wujiaju/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16
log=/data/wujiaju/logs/graph_path_fixed_h1_j_20260801/inverse_reuse_10k_balanced_jcalls_r64_s16_gpu${CUDA_VISIBLE_DEVICES}.log

mkdir -p "$out_dir" "$(dirname "$log")"
cd "$repo_dir"

"$python_bin" -u -m reasoning_loop.train_graph_path_age_specific_j_bank \
  --checkpoint "$checkpoint" \
  --phase-summary "$phase_summary" \
  --out-dir "$out_dir" \
  --rank 64 \
  --stage-rank 16 \
  --map-architecture shared_diagonal_stage_lora \
  --initialization age_specific_bank \
  --bank-init-artifact "$source_bank" \
  --curriculum inverse_reuse \
  --rollback-composition product \
  --fixed-start-age 1 \
  --lr-schedule wsd \
  --warmup-fraction 0.1 \
  --warmup-start-factor 0.1 \
  --decay-fraction 0.1 \
  --decay-end-factor 0.1 \
  --diagonal-lr-multiplier 1.0 \
  --cuda-memory-fraction 0.02 \
  --seed 826101 \
  >"$log" 2>&1

"$python_bin" - "$out_dir/summary.json" "$out_dir/age_specific_j_bank_after_inverse_reuse_10k.pt" <<'PY'
import collections
import csv
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
artifact = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader((artifact.parent / "training_trajectories.csv").open()))
counts = collections.Counter(int(row["back_count"]) for row in rows)
assert summary["status"] == "complete"
assert summary["training_start_age"] == 1
assert summary["training_trajectory_count"] == 10_000
assert counts == {1: 4379, 2: 2190, 3: 1460, 4: 1095, 5: 876}
j_calls = collections.Counter()
for row in rows:
    j_calls.update(int(value) for value in row["rollback_sources"].split(",") if value)
assert sum(j_calls.values()) == 21_899
assert max(j_calls.values()) - min(j_calls.values()) <= 1
assert artifact.is_file() and artifact.stat().st_size > 0
print(json.dumps({"status": "validated", "reuse_counts": counts, "j_calls": j_calls}, sort_keys=True))
PY
