#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731/architecture_round_20260803
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
FINAL_ROOT="${RUN_ROOT}/finalizer"
MANIFEST_PATH="${FINAL_ROOT}/manifest.json"
mkdir -p "${FINAL_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${CODE_DIR}"
export MPLBACKEND=Agg
cd "${CODE_DIR}"

write_status() {
    local status="$1" phase="$2" detail="${3:-}"
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${phase}" "${detail}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {"created_unix": time.time()}
data.update({"status": sys.argv[2], "phase": sys.argv[3],
             "detail": sys.argv[4] or None, "updated_unix": time.time()})
if sys.argv[2] in {"complete", "failed"}: data["finished_unix"] = time.time()
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

wait_complete() {
    local manifest="$1" label="$2"
    while true; do
        state="$("${PYTHON_BIN}" - "${manifest}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(json.loads(p.read_text()).get("status", "missing") if p.exists() else "missing")
PY
)"
        if [[ "${state}" == complete ]]; then return 0; fi
        if [[ "${state}" == failed ]]; then
            write_status failed "wait_${label}" "${manifest} reported failed"
            return 1
        fi
        sleep 30
    done
}

write_status running wait_architecture_shard1
wait_complete "${RUN_ROOT}/shard1/pipeline_manifest.json" architecture_shard1

write_status running screen_architecture_shard1
bash scripts/run_addition_balanced_carry_screen_20260803.sh 4 1 \
    > "${LOG_ROOT}/addition_balanced_carry_screen_shard1.log" 2>&1

write_status running aggregate_round2
"${PYTHON_BIN}" -m scripts.aggregate_addition_architecture_round \
    --round-root "${RUN_ROOT}" \
    --reference-root /data/wujiaju/paper_length_telomere_20260731/checkpoint_sweep_20260803 \
    --out-dir "${RUN_ROOT}/aggregate" \
    > "${LOG_ROOT}/addition_architecture_round_aggregate.log" 2>&1
"${PYTHON_BIN}" -m scripts.aggregate_addition_balanced_carry_screen \
    --screen-root "${RUN_ROOT}/balanced_carry_screen" \
    --id-screen-root "${RUN_ROOT}/balanced_carry_id1to10_screen" \
    --out-dir "${RUN_ROOT}/balanced_carry_screen/aggregate" \
    --threshold 0.98 \
    > "${LOG_ROOT}/addition_balanced_carry_screen_aggregate.log" 2>&1

# Let the already-running 10k/40k reference controller queues finish before
# refreshing the shards with the final architecture eligibility list.
write_status running wait_reference_j_shards
wait_complete "${RUN_ROOT}/carry_j_round/shard0_manifest.json" reference_j_shard0
wait_complete "${RUN_ROOT}/carry_j_round/shard1_manifest.json" reference_j_shard1

write_status running final_j_shards
set +e
bash scripts/run_addition_architecture_j_round_20260803.sh 0 0 \
    > "${LOG_ROOT}/addition_architecture_carry_j_final_shard0.log" 2>&1 &
PID0=$!
bash scripts/run_addition_architecture_j_round_20260803.sh 6 1 \
    > "${LOG_ROOT}/addition_architecture_carry_j_final_shard1.log" 2>&1 &
PID1=$!
wait "${PID0}"; STATUS0=$?
wait "${PID1}"; STATUS1=$?
set -e
if [[ "${STATUS0}" -ne 0 ]]; then
    write_status failed final_j_shard0 "exit ${STATUS0}"
    exit "${STATUS0}"
fi
if [[ "${STATUS1}" -eq 75 ]]; then
    write_status running retry_final_j_shard1_on_gpu0
    bash scripts/run_addition_architecture_j_round_20260803.sh 0 1 \
        >> "${LOG_ROOT}/addition_architecture_carry_j_final_shard1.log" 2>&1
elif [[ "${STATUS1}" -ne 0 ]]; then
    write_status failed final_j_shard1 "exit ${STATUS1}"
    exit "${STATUS1}"
fi

write_status running aggregate_j
"${PYTHON_BIN}" -m scripts.aggregate_addition_architecture_j_round \
    --j-root "${RUN_ROOT}/carry_j_round" \
    --out-dir "${RUN_ROOT}/carry_j_round/aggregate" \
    > "${LOG_ROOT}/addition_architecture_carry_j_aggregate.log" 2>&1

write_status running spectrum_analysis
for CONTROLLER in "${RUN_ROOT}"/carry_j_round/*/controller.pt; do
    [[ -f "${CONTROLLER}" ]] || continue
    JOB="$(basename "$(dirname "${CONTROLLER}")")"
    "${PYTHON_BIN}" -m scripts.analyze_paper_controller_spectrum \
        --artifact "${CONTROLLER}" --label "${JOB}" \
        --out-dir "${RUN_ROOT}/carry_j_round/${JOB}/spectrum" \
        > "${LOG_ROOT}/addition_architecture_carry_j_spectrum_${JOB}.log" 2>&1
done
"${PYTHON_BIN}" -m scripts.aggregate_paper_controller_spectra \
    --root "${RUN_ROOT}/carry_j_round" \
    --out-dir "${RUN_ROOT}/carry_j_round/aggregate_spectra" \
    > "${LOG_ROOT}/addition_architecture_carry_j_spectra_aggregate.log" 2>&1
"${PYTHON_BIN}" -m scripts.compose_addition_architecture_final_report \
    --experiment-root "${RUN_ROOT}" \
    --checkpoint-root /data/wujiaju/paper_length_telomere_20260731/checkpoint_sweep_20260803/aggregate \
    --out-path "${RUN_ROOT}/FINAL_REPORT.md" \
    > "${LOG_ROOT}/addition_architecture_final_report.log" 2>&1

write_status complete done
