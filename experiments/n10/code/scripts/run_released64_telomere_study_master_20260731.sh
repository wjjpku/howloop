#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/paperexperiment/paper_length_telomere_20260731
WAITER=/data/paperexperiment/LooPlus/scripts/wait_for_empty_gpu_paper_length_telomere_20260731.sh
PIPELINE=/data/paperexperiment/LooPlus/scripts/run_released64_telomere_seed_pipeline_20260731.sh
BENCHMARK_MANIFEST="${RUN_ROOT}/manifests/parity_adaptive_step_released64_seed0/released_benchmark_fp32.json"
BENCHMARK_SUMMARY="${RUN_ROOT}/backbones/parity_adaptive_step_released64_seed0_benchmark_fp32/summary.json"

export BASELINE_VARIANT=released64

if ! [[ -f "${BENCHMARK_MANIFEST}" ]] || ! grep -q '"status": "complete"' "${BENCHMARK_MANIFEST}"; then
    bash "${WAITER}" released_benchmark_fp32 parity adaptive_step 0
fi

/data/paperexperiment/.venvs/loopreasoner/bin/python - "${BENCHMARK_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "status": payload.get("status") == "complete",
    "steps": payload.get("steps") == 200,
    "heads": payload.get("model", {}).get("n_heads") == 64,
    "precision": payload.get("training_precision") == "fp32",
    "supervision": payload.get("supervision") == "adaptive_step",
    "peak_recorded": payload.get("peak_cuda_memory_reserved_gib", 0) > 0,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"released FP32 benchmark failed invariants: {failed}")
PY

for SEED in 0 1 2; do
    SESSION="paper_tel_released64_seed${SEED}_pipeline"
    LOG="${RUN_ROOT}/queue/released64_seed${SEED}_pipeline.log"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" \
            "bash ${PIPELINE} ${SEED} >> ${LOG} 2>&1"
    fi
done

echo "$(date -Is) released64 FP32 benchmark verified; seed pipelines dispatched"
