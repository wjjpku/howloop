#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
root=/data/wujiaju/graph_path_age_specific_j_bank_20260801
run=${root}/seed0_full_affine_two_axis_curriculum
validation=${run}/validation_ce_checkpoints

mkdir -p "${validation}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

labels=(single_back T04_R1 T08_R1 T08_R2 T12_R2 T12_R3 T16_R3 T16_R4 T20_R4 T20_R5 T24_R5 T24_R6 T28_R6 T28_R7)
artifacts=(
  "${root}/seed0_full_affine_single_back/age_specific_j_bank.pt"
  "${run}/age_specific_j_bank_after_T04_R1.pt"
  "${run}/age_specific_j_bank_after_T08_R1.pt"
  "${run}/age_specific_j_bank_after_T08_R2.pt"
  "${run}/age_specific_j_bank_after_T12_R2.pt"
  "${run}/age_specific_j_bank_after_T12_R3.pt"
  "${run}/age_specific_j_bank_after_T16_R3.pt"
  "${run}/age_specific_j_bank_after_T16_R4.pt"
  "${run}/age_specific_j_bank_after_T20_R4.pt"
  "${run}/age_specific_j_bank_after_T20_R5.pt"
  "${run}/age_specific_j_bank_after_T24_R5.pt"
  "${run}/age_specific_j_bank_after_T24_R6.pt"
  "${run}/age_specific_j_bank_after_T28_R6.pt"
  "${run}/age_specific_j_bank_after_T28_R7.pt"
)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  "${python_bin}" -m reasoning_loop.audit_graph_path_age_specific_j_bank \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --bank-artifact "${artifacts[$index]}" \
    --out-dir "${validation}/${label}" \
    --device cuda \
    --seed 970001 \
    --trajectories 28 \
    --batch-size 64 \
    --train-max-backs 28 \
    --unseen-max-backs 40 \
    --composition-examples 64 \
    --composition-batch-size 64 \
    --conditions learned \
    > "${validation}/${label}.log" 2>&1
done

"${python_bin}" - "${validation}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    row = {"checkpoint": path.parent.name}
    for item in payload["summary"]:
        row[f"{item['split']}_accuracy"] = item["accuracy_mean"]
        row[f"{item['split']}_ce"] = item["cross_entropy_mean"]
    row["mean_validation_ce"] = (
        row["train_like_ce"] + row["longer_unseen_ce"]
    ) / 2
    rows.append(row)
rows.sort(key=lambda row: row["mean_validation_ce"])
with (root / "checkpoint_validation_ce.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "best_checkpoint.json").write_text(
    json.dumps(rows[0], indent=2, sort_keys=True) + "\n"
)
print(json.dumps(rows, indent=2, sort_keys=True))
PY
