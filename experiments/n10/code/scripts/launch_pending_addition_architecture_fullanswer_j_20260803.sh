#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus
PYTHON_BIN=/data/wujiaju/.venvs/loopreasoner/bin/python
REDESIGN_ROOT=/data/wujiaju/paper_length_telomere_20260731/full_answer_redesign_20260803
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731/fullanswer_architecture_j
mkdir -p "${LOG_ROOT}"

wait_complete() {
    local manifest="$1" label="$2" state
    while true; do
        state="$("${PYTHON_BIN}" - "${manifest}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(json.loads(p.read_text()).get("status", "missing") if p.is_file() else "missing")
PY
)"
        if [[ "${state}" == complete ]]; then return 0; fi
        if [[ "${state}" == failed ]]; then
            echo "${label} failed: ${manifest}" >&2
            return 1
        fi
        sleep 30
    done
}

launch_shard() {
    local shard="$1" gpu="$2" session="addition_arch_fullanswer_j_s${1}_20260803"
    if tmux has-session -t "${session}" 2>/dev/null; then return 0; fi
    tmux new-session -d -s "${session}" \
        "bash ${CODE_DIR}/scripts/run_addition_architecture_fullanswer_j_20260803.sh ${gpu} ${shard} >> ${LOG_ROOT}/shard${shard}_pipeline.log 2>&1"
}

wait_complete \
    "${REDESIGN_ROOT}/armA_fixed_n10_40k_seed0/pipeline_manifest.json" armA
launch_shard 1 6

wait_complete \
    "${REDESIGN_ROOT}/armB_adaptive_n1to10_40k_seed0/pipeline_manifest.json" armB
launch_shard 2 5
