#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${GPU_ID:?GPU_ID is required}"
: "${SOURCE_ID:?SOURCE_ID is required}"
: "${BASE_CODE:?BASE_CODE is required}"
: "${ARM_KIND:?ARM_KIND must be affine or resffn128}"
: "${CURRICULUM_SCRIPT:?CURRICULUM_SCRIPT is required}"

PYTHON_BIN="${PYTHON_BIN:-/data/wujiaju/.venvs/loopreasoner/bin/python}"
BACKBONE="/data/wujiaju/kg-fj-nope-20260815-v1/nope_only/backbone/best.pt"
BACKBONE_SHA="4d6bb56b501ddeb0b1056945fd41f6a5f7ce1b79651732eaef077b007b2641c5"
ORACLE="/data/wujiaju/kg-fj-length-20260814-v4/run/oracle/best.pt"
ORACLE_SHA="db8451045025ab110fbe246a48297cc81c175e9837e4fac1e83de1087ddd4a34"

case "$ARM_KIND" in
  affine)
    CONTROLLER_ARCH="affine"
    HIDDEN_WIDTH=128
    EXPECTED_PARAMETERS=65792
    ;;
  resffn128)
    CONTROLLER_ARCH="mlp"
    HIDDEN_WIDTH=128
    EXPECTED_PARAMETERS=66432
    ;;
  *)
    echo "unknown ARM_KIND: $ARM_KIND" >&2
    exit 2
    ;;
esac

case "$RUN_ROOT" in /data/wujiaju/*) ;; *) exit 2 ;; esac
case "$LOG_ROOT" in /data/wujiaju/logs/*) ;; *) exit 2 ;; esac
case "$BASE_CODE" in /data/wujiaju/*) ;; *) exit 2 ;; esac
if [[ -e "$RUN_ROOT" || -e "$LOG_ROOT" ]]; then
  echo "run or log root already exists" >&2
  exit 3
fi

mkdir "$RUN_ROOT"
mkdir -p "$LOG_ROOT"
"$PYTHON_BIN" - "$RUN_ROOT/RUNNING.json" "$SOURCE_ID" "$GPU_ID" \
  "$ARM_KIND" "$EXPECTED_PARAMETERS" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema": "kg-fj-matched-controller-m64-v1",
    "status": "running",
    "source_id": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "arm": sys.argv[4],
    "expected_controller_parameters": int(sys.argv[5]),
    "initial_training_lengths": [4, 5, 6],
    "initial_training_steps": 50000,
    "curriculum_stages": [20, 24, 28, 32, 40, 48, 56, 64],
    "max_steps_per_stage": 12000,
    "strict_ood_lengths": list(range(65, 73)),
}, sort_keys=True) + "\n")
PY

cd "$BASE_CODE"
export PYTHONPATH="$BASE_CODE"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

"$PYTHON_BIN" - "$CONTROLLER_ARCH" "$HIDDEN_WIDTH" "$EXPECTED_PARAMETERS" <<'PY'
import sys
from experiments.kg_fj_length.controller import ControllerConfig, build_controller
arch, width, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
controller = build_controller(ControllerConfig(
    d_model=256,
    hidden_width=width,
    initial_scale=1.0e-2,
    architecture=arch,
    attention_heads=8,
), seed=301)
actual = sum(parameter.numel() for parameter in controller.parameters())
if actual != expected:
    raise RuntimeError(f"controller parameter mismatch: {actual} != {expected}")
PY

"$PYTHON_BIN" -u -m experiments.kg_fj_length.cli controller \
  --output-dir "$RUN_ROOT/initial_controller" \
  --device cuda:0 \
  --seed 301 \
  --world-seed 7 \
  --source-commit "$SOURCE_ID" \
  --steps 50000 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --warmup-steps 1000 \
  --stable-steps 39000 \
  --decay-steps 10000 \
  --eval-interval 1000 \
  --selection-count 512 \
  --test-count 4096 \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --hidden-width "$HIDDEN_WIDTH" \
  --initial-scale 1e-2 \
  --controller-architecture "$CONTROLLER_ARCH" \
  --attention-heads 8 \
  --controller-loss-mode truncated_unroll \
  >"$LOG_ROOT/initial_controller.log" 2>&1

INITIAL_CONTROLLER_SHA=$(sha256sum "$RUN_ROOT/initial_controller/best.pt" | cut -d' ' -f1)
"$PYTHON_BIN" -u "$CURRICULUM_SCRIPT" \
  --output-dir "$RUN_ROOT/curriculum" \
  --device cuda:0 \
  --source-commit "$SOURCE_ID" \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --initial-controller-checkpoint "$RUN_ROOT/initial_controller/best.pt" \
  --initial-controller-sha256 "$INITIAL_CONTROLLER_SHA" \
  --stages 20,24,28,32,40,48,56,64 \
  --seed 601 \
  --batch-size 256 \
  --max-steps-per-stage 12000 \
  --min-steps-per-stage 4000 \
  --eval-interval 1000 \
  --learning-rate 5e-5 \
  --warmup-steps 500 \
  --stable-steps 9000 \
  --decay-steps 2500 \
  --current-length-probability 0.5 \
  --stage-accuracy-threshold 0.98 \
  --retention-accuracy-threshold 0.98 \
  --selection-count 256 \
  --test-count 1024 \
  >"$LOG_ROOT/curriculum.log" 2>&1

CONTROLLER_SHA=$(sha256sum "$RUN_ROOT/curriculum/best.pt" | cut -d' ' -f1)
"$PYTHON_BIN" -m experiments.kg_fj_length.cli evaluate \
  --output-dir "$RUN_ROOT/evaluate" \
  --device cuda:0 \
  --seed 20260818 \
  --count 1024 \
  --min-length 1 \
  --max-length 72 \
  --adaptive-extra-calls 4 \
  --source-commit "$SOURCE_ID" \
  --backbone-checkpoint "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --controller-checkpoint "$RUN_ROOT/curriculum/best.pt" \
  --controller-sha256 "$CONTROLLER_SHA" \
  --oracle-checkpoint "$ORACLE" \
  --oracle-sha256 "$ORACLE_SHA" \
  >"$LOG_ROOT/evaluate.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_ID" "$GPU_ID" "$ARM_KIND" \
  "$INITIAL_CONTROLLER_SHA" "$CONTROLLER_SHA" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
evaluation = root / "evaluate" / "summary.json"
payload = {
    "schema": "kg-fj-matched-controller-m64-v1",
    "status": "complete",
    "source_id": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "arm": sys.argv[4],
    "initial_controller_sha256": sys.argv[5],
    "final_controller_sha256": sys.argv[6],
    "evaluation_summary_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
    "trained_max_length": 64,
    "strict_ood_lengths": list(range(65, 73)),
}
(root / "PIPELINE_COMPLETE.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
PY
