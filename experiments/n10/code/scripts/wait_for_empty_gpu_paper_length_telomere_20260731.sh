#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "usage: $0 ACTION TASK SUPERVISION [SEED]" >&2
    exit 2
fi

ACTION="$1"
TASK="$2"
SUPERVISION="$3"
SEED="${4:-0}"
RUNNER=/data/wujiaju/LooPlus/scripts/run_paper_length_telomere_remote_20260731.sh
QUEUE_ROOT=/data/wujiaju/paper_length_telomere_20260731/queue
LOCK_ROOT=/data/wujiaju/paper_length_telomere_20260731/locks
mkdir -p "${QUEUE_ROOT}" "${LOCK_ROOT}"

while true; do
    # Prefer the originally preflighted cards, then consider every other card.
    # The runner still requires a truly empty device (no compute PID and at
    # most 128 MiB reported used), so widening the poll set never co-locates.
    for SLOT in 0 1 2; do
        for GPU in 1 7 0 2 3 4 5 6; do
            set +e
            (
                # Enforce the shared-devbox limit of at most three paper-study
                # GPU jobs even when several task managers are waiting.
                flock -n 8 || exit 75
                flock -n 9 || exit 75
                USED="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
                PIDS="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
                if [[ -n "${PIDS}" || "${USED}" -gt 128 ]]; then
                    exit 75
                fi
                echo "$(date -Is) launching action=${ACTION} task=${TASK} supervision=${SUPERVISION} seed=${SEED} gpu=${GPU} slot=${SLOT}"
                bash "${RUNNER}" "${ACTION}" "${TASK}" "${SUPERVISION}" "${GPU}" "${SEED}"
            ) 8>"${LOCK_ROOT}/paper_job_slot${SLOT}.lock" 9>"${LOCK_ROOT}/gpu${GPU}.lock"
            STATUS=$?
            set -e
            if [[ "${STATUS}" -eq 0 ]]; then
                echo "$(date -Is) completed action=${ACTION} task=${TASK} supervision=${SUPERVISION} seed=${SEED}"
                exit 0
            fi
            if [[ "${STATUS}" -ne 75 ]]; then
                echo "$(date -Is) failed action=${ACTION} task=${TASK} supervision=${SUPERVISION} seed=${SEED} exit=${STATUS}" >&2
                exit "${STATUS}"
            fi
        done
    done
    echo "$(date -Is) waiting action=${ACTION} task=${TASK} supervision=${SUPERVISION} seed=${SEED}; all GPUs busy"
    sleep 30
done
