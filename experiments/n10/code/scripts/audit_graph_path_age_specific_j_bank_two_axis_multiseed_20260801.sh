#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
root=/data/paperexperiment/graph_path_age_specific_j_bank_20260801
out=${root}/two_axis_multiseed_fixed_test

mkdir -p "${out}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

labels=(trainseed820001 trainseed820002 trainseed820003)
artifacts=(
  "${root}/seed0_full_affine_two_axis_curriculum/age_specific_j_bank.pt"
  "${root}/seed0_full_affine_two_axis_trainseed820002/age_specific_j_bank.pt"
  "${root}/seed0_full_affine_two_axis_trainseed820003/age_specific_j_bank.pt"
)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  "${python_bin}" -m reasoning_loop.audit_graph_path_age_specific_j_bank \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --bank-artifact "${artifacts[$index]}" \
    --out-dir "${out}/${label}" \
    --device cuda \
    --seed 980001 \
    --trajectories 112 \
    --batch-size 64 \
    --train-max-backs 28 \
    --unseen-max-backs 40 \
    --composition-examples 64 \
    --composition-batch-size 64 \
    --conditions learned wrong_stage shared_J8 \
    > "${out}/${label}.log" 2>&1
done

"${python_bin}" - "${out}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    for item in payload["summary"]:
        rows.append({"training_seed": path.parent.name, **item})
with (root / "per_seed_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["split"], row["condition"])].append(row)
aggregate = []
for (split, condition), parts in sorted(groups.items()):
    result = {"split": split, "condition": condition, "training_seeds": len(parts)}
    for field in ("accuracy_mean", "cross_entropy_mean"):
        values = [float(part[field]) for part in parts]
        result[field] = statistics.mean(values)
        result[field + "_std"] = statistics.stdev(values)
        result[field + "_min"] = min(values)
        result[field + "_max"] = max(values)
    aggregate.append(result)
with (root / "aggregate_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
    writer.writeheader()
    writer.writerows(aggregate)
print(json.dumps(aggregate, indent=2, sort_keys=True))
PY
