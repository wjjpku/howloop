#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-6}"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
root=/data/wujiaju/graph_path_age_specific_j_bank_20260801
out=${root}/single_back_matched_audit

mkdir -p "${out}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

labels=(rank96_init full_affine_single_back)
artifacts=(
  "${root}/seed0_rank96_final_ce_canonical_init/age_specific_j_bank.pt"
  "${root}/seed0_full_affine_single_back/age_specific_j_bank.pt"
)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  "${python_bin}" -m reasoning_loop.audit_graph_path_age_specific_j_bank \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --bank-artifact "${artifacts[$index]}" \
    --out-dir "${out}/${label}" \
    --device cuda \
    --seed 960001 \
    --trajectories 56 \
    --batch-size 64 \
    --train-max-total-backs 1 \
    --unseen-max-total-backs 1 \
    --max-consecutive-backs 1 \
    --composition-examples 64 \
    --composition-batch-size 64 \
    > "${out}/${label}.log" 2>&1
done

"${python_bin}" - "${out}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    for item in payload["summary"]:
        rows.append({"model": path.parent.name, **item})
with (root / "matched_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2, sort_keys=True))
PY
