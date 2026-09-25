#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/reverse_addition_20260804
BACKBONE_LABEL=addition_lsb_fixed_n10_t11_posabs_seed0
CHECKPOINT="${RUN_ROOT}/backbones/${BACKBONE_LABEL}/selected.pt"
LABEL="${BACKBONE_LABEL}_rank48_identity_fullanswer_l1to20_anchor1_wsd5376_seed522001"
CONTROLLER_DIR="${RUN_ROOT}/controllers/${LABEL}"
AUDIT_DIR="${RUN_ROOT}/audits/${LABEL}_l1to30"
HEATMAP_DIR="${RUN_ROOT}/audits/${LABEL}_length_loop_accuracy_n1to30_t1to35"
MATRIX_DIR="${RUN_ROOT}/controllers/${LABEL}_matrix_analysis"
MANIFEST="${RUN_ROOT}/controllers/${LABEL}_manifest.json"
PYTHON=/data/wujiaju/.venvs/loopreasoner/bin/python

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing checkpoint: ${CHECKPOINT}" >&2
    exit 3
fi
for path in "${CONTROLLER_DIR}" "${AUDIT_DIR}" "${HEATMAP_DIR}" "${MATRIX_DIR}"; do
    if [[ -e "${path}" ]]; then
        echo "refusing to overwrite existing output: ${path}" >&2
        exit 4
    fi
done

cd /data/wujiaju/LooPlus
"${PYTHON}" - "${MANIFEST}" "${CHECKPOINT}" "${CONTROLLER_DIR}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "status": "running",
    "started_at": datetime.now().astimezone().isoformat(),
    "checkpoint": sys.argv[2],
    "controller_dir": sys.argv[3],
    "controller_train_lengths": [1, 20],
    "target_rule": "T(n)=n+1",
    "supervision": "final-only full answer-region CE including final carry",
    "controller": "row-vector J(h)=h@diag(D)+(h@A)@B+b, rank 48, identity initialization",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"${PYTHON}" -u -m reasoning_loop.paper_length_telomere controller \
    --checkpoint "${CHECKPOINT}" \
    --controller-parameterization diagonal_low_rank \
    --rank 48 \
    --seed 522001 \
    --device cuda \
    --grad-clip 1.0 \
    --learning-rate-multiplier 5.0 \
    --diagonal-lr-multiplier 0.1 \
    --dense-stage-count 0 \
    --controller-training-profile identity_long_warmup \
    --controller-initialization identity \
    --controller-curriculum logical_range \
    --controller-logical-min-length 1 \
    --controller-logical-max-length 20 \
    --controller-anchor-step 1 \
    --controller-warmup-updates 2048 \
    --controller-stable-updates 2816 \
    --controller-final-lr-ratio 0.1 \
    --controller-lr-schedule wsd \
    --controller-ce-temperature 1.0 \
    --controller-supervision full_answer \
    --stage-round-multiplier 3 \
    --controller-checkpoint-every 256 \
    --no-controller-post-final-j \
    --force \
    --out-dir "${CONTROLLER_DIR}"

"${PYTHON}" -u -m reasoning_loop.paper_length_telomere audit \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER_DIR}/controller.pt" \
    --lengths {1..30} \
    --batch-size 128 \
    --batches 8 \
    --maximum-step 35 \
    --modes full no_AB identity_D \
    --seed 884101 \
    --device cuda \
    --no-post-final-j \
    --out-dir "${AUDIT_DIR}"

"${PYTHON}" -u -m scripts.plot_addition_length_loop_accuracy \
    --checkpoint "${CHECKPOINT}" \
    --controller "${CONTROLLER_DIR}/controller.pt" \
    --maximum-length 30 \
    --maximum-step 35 \
    --examples 1024 \
    --batch-size 128 \
    --seed 884101 \
    --device cuda \
    --out-dir "${HEATMAP_DIR}"

"${PYTHON}" -u -m scripts.analyze_addition_diag_lora_controllers \
    --controller "J_train_1to20=${CONTROLLER_DIR}/controller.pt" \
    --out-dir "${MATRIX_DIR}"

"${PYTHON}" - "${MANIFEST}" "${AUDIT_DIR}" "${HEATMAP_DIR}" "${MATRIX_DIR}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "status": "complete",
    "completed_at": datetime.now().astimezone().isoformat(),
    "audit_dir": sys.argv[2],
    "heatmap_dir": sys.argv[3],
    "matrix_dir": sys.argv[4],
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "COMPLETE ${LABEL}"
