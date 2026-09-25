#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/data/wujiaju/LooPlus
RUN_ROOT=/data/wujiaju/paper_length_telomere_20260731
LOG_ROOT=/data/wujiaju/logs/paper_length_telomere_20260731
QUEUE_LOG="${LOG_ROOT}/fixed_n10_sumreverse_wave_manager.log"
RUNNER="${CODE_DIR}/scripts/run_fixed_n10_baseline_20260803.sh"
REQUIRED_FREE_MIB=22528
RESERVE_MIB=16384

mkdir -p "${LOG_ROOT}"
exec >> "${QUEUE_LOG}" 2>&1

manifest_status() {
    local task="$1" seed="$2" loops=10
    if [[ "${task}" == "addition" ]]; then loops=11; fi
    local path="${RUN_ROOT}/backbones/${task}_fixed_n10_t${loops}_official_seed${seed}/pipeline_manifest.json"
    if [[ ! -f "${path}" ]]; then
        printf 'missing'
        return
    fi
    python3 - "${path}" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("status", "unknown"))
PY
}

wait_for_addition_wave() {
    while true; do
        local terminal=1
        for seed in 0 1 2; do
            local status
            status="$(manifest_status addition "${seed}")"
            if [[ "${status}" == "failed" ]]; then
                echo "$(date -Is) addition seed${seed} failed; refusing Sum-Reverse wave"
                exit 3
            fi
            if [[ "${status}" != "complete" ]]; then terminal=0; fi
        done
        if [[ "${terminal}" -eq 1 ]]; then return; fi
        echo "$(date -Is) waiting for all Addition fixed-n10 baselines"
        sleep 60
    done
}

snapshot_gpus() {
    local tag="$1"
    shift
    echo "$(date -Is) snapshot=${tag} gpus=$*"
    for gpu in "$@"; do
        nvidia-smi -i "${gpu}" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
        local pids
        pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
        for pid in ${pids}; do
            ps -o user=,pid=,etime=,cmd= -p "${pid}" || true
            nvidia-smi -i "${gpu}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^${pid}," || true
        done
    done
}

double_preflight() {
    local gpus=("$@")
    local first_used=() first_free=()
    snapshot_gpus first "${gpus[@]}"
    for gpu in "${gpus[@]}"; do
        first_used+=("$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')")
        first_free+=("$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')")
    done
    sleep 60
    snapshot_gpus second "${gpus[@]}"
    for index in "${!gpus[@]}"; do
        local gpu="${gpus[$index]}" used free delta
        used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        delta=$((used - first_used[$index]))
        if [[ "${delta}" -lt 0 ]]; then delta=$((-delta)); fi
        if [[ "${first_free[$index]}" -lt "${REQUIRED_FREE_MIB}" ]] || \
           [[ "${free}" -lt "${REQUIRED_FREE_MIB}" ]] || [[ "${delta}" -gt 512 ]]; then
            echo "$(date -Is) GPU${gpu} failed double preflight: first_free=${first_free[$index]} second_free=${free} used_delta=${delta}"
            exit 75
        fi
    done
}

launch_sum() {
    local gpu="$1"
    local seed="$2"
    local session="sum_reverse_fixed_n10_seed${seed}_20260803"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "$(date -Is) session already exists: ${session}"
        return
    fi
    echo "$(date -Is) launching Sum-Reverse seed${seed} on GPU${gpu}"
    tmux new-session -d -s "${session}" \
        "cd ${CODE_DIR} && exec bash ${RUNNER} sum_reverse ${gpu} ${seed}"
}

monitor_first_sum() {
    local path="${RUN_ROOT}/backbones/sum_reverse_fixed_n10_t10_official_seed0/pipeline_manifest.json"
    for _ in $(seq 1 10); do
        sleep 30
        if [[ ! -f "${path}" ]]; then continue; fi
        local status free
        status="$(manifest_status sum_reverse 0)"
        free="$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        echo "$(date -Is) Sum-Reverse seed0 status=${status} GPU4_free=${free}"
        if [[ "${status}" == "failed" ]] || [[ "${free}" -lt "${RESERVE_MIB}" ]]; then
            echo "$(date -Is) first Sum-Reverse launch failed safety monitoring"
            exit 76
        fi
        if [[ "${status}" == "complete" ]]; then return; fi
    done
}

echo "$(date -Is) Sum-Reverse fixed-n10 wave manager started"
wait_for_addition_wave
echo "$(date -Is) Addition wave complete"
double_preflight 4
launch_sum 4 0
monitor_first_sum
double_preflight 5 0
launch_sum 5 1
launch_sum 0 2
echo "$(date -Is) Sum-Reverse fixed-n10 wave fully launched"
