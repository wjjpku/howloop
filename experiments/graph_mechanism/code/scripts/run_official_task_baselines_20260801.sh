#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 TASK" >&2
    exit 2
fi

TASK="$1"
case "${TASK}" in
    copy)
        EXPECTED_LAYERS=2
        EXPECTED_HEADS=8
        EXPECTED_TRAIN_MAX=19
        ;;
    addition)
        EXPECTED_LAYERS=3
        EXPECTED_HEADS=8
        EXPECTED_TRAIN_MAX=19
        ;;
    sum_reverse)
        EXPECTED_LAYERS=2
        EXPECTED_HEADS=16
        EXPECTED_TRAIN_MAX=19
        ;;
    *)
        echo "TASK must be one of: copy, addition, sum_reverse" >&2
        exit 2
        ;;
esac

RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
WAITER=/data/wujiaju/LooPlus/scripts/wait_for_empty_gpu_paper_length_telomere_20260731.sh
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
LABEL="${TASK}_adaptive_step_official_seed0"
BENCHMARK_MANIFEST="${RUN_ROOT}/manifests/${LABEL}/official_benchmark_fp32.json"
BENCHMARK_SUMMARY="${RUN_ROOT}/backbones/${LABEL}_benchmark_fp32/summary.json"
TASK_LOCK="${RUN_ROOT}/locks/official_${TASK}_baseline_manager.lock"
mkdir -p "${RUN_ROOT}/locks"

# The all-task and task-specific masters may both request this task.  Holding
# the task lock prevents duplicate training of the same checkpoint.
exec 7>"${TASK_LOCK}"
flock 7

export BASELINE_VARIANT=official
if ! [[ -f "${BENCHMARK_MANIFEST}" ]] || ! grep -q '"status": "complete"' "${BENCHMARK_MANIFEST}"; then
    bash "${WAITER}" official_benchmark_fp32 "${TASK}" adaptive_step 0
fi

"${PYTHON_BIN}" - "${BENCHMARK_SUMMARY}" "${TASK}" "${EXPECTED_LAYERS}" "${EXPECTED_HEADS}" "${EXPECTED_TRAIN_MAX}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
expected_layers = int(sys.argv[3])
expected_heads = int(sys.argv[4])
expected_train_max = int(sys.argv[5])
payload = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "status": payload.get("status") == "complete",
    "steps": payload.get("steps") == 200,
    "task": payload.get("task", {}).get("name") == task,
    "train_maximum": payload.get("task", {}).get("train_max_length") == expected_train_max,
    "layers": payload.get("model", {}).get("block_layers") == expected_layers,
    "heads": payload.get("model", {}).get("n_heads") == expected_heads,
    "width": payload.get("model", {}).get("d_model") == 256,
    "mlp": payload.get("model", {}).get("d_mlp") == 1024,
    "official": payload.get("official_model_config") is True,
    "precision": payload.get("training_precision") == "fp32",
    "supervision": payload.get("supervision") == "adaptive_step",
    "loss": payload.get("loss_placement") == "answer-region CE at each sample's T(n) only",
    "peak_recorded": payload.get("peak_cuda_memory_reserved_gib", 0) > 0,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"official {task} FP32 benchmark failed invariants: {failed}")
PY

for SEED in 0 1 2; do
    FORMAL_LABEL="${TASK}_adaptive_step_official_seed${SEED}"
    FORMAL_MANIFEST="${RUN_ROOT}/manifests/${FORMAL_LABEL}/official_formal.json"
    if [[ -f "${FORMAL_MANIFEST}" ]] && grep -q '"status": "complete"' "${FORMAL_MANIFEST}"; then
        echo "$(date -Is) official baseline already complete task=${TASK} seed=${SEED}"
        continue
    fi
    bash "${WAITER}" official_formal "${TASK}" adaptive_step "${SEED}"
done

echo "$(date -Is) all official baseline seeds complete task=${TASK} seeds=0,1,2"
