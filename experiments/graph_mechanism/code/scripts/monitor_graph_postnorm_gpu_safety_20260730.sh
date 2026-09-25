#!/usr/bin/env bash
set -euo pipefail

log_path="/data/wujiaju/logs/graph_path_postnorm_D8L8_20260730/safety_monitor.log"
reserve_mib=16384

mkdir -p "$(dirname "${log_path}")"
echo "$(date --iso-8601=seconds) MONITOR_START gpus=1,3,4 reserve_mib=${reserve_mib}" \
  >>"${log_path}"

while true; do
  active=0
  for session in graph_postnorm_q0 graph_postnorm_q1 graph_postnorm_q2; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      active=1
    fi
  done
  if (( active == 0 )); then
    echo "$(date --iso-8601=seconds) MONITOR_COMPLETE no_active_queues" \
      >>"${log_path}"
    exit 0
  fi

  for gpu in 1 3 4; do
    free_mib="$(
      nvidia-smi \
        -i "${gpu}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
        tr -d ' '
    )"
    echo "$(date --iso-8601=seconds) gpu=${gpu} free_mib=${free_mib}" \
      >>"${log_path}"
    if (( free_mib >= reserve_mib )); then
      continue
    fi
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      owner="$(ps -o user= -p "${pid}" | xargs)"
      command="$(ps -o cmd= -p "${pid}")"
      if [[ "${owner}" == "wujiaju" ]] &&
        [[ "${command}" == *"reasoning_loop.graph_path_loop"* ]] &&
        [[ "${command}" == *"graph_path_postnorm_D8L8_20260730"* ]]; then
        echo "$(date --iso-8601=seconds) STOP_OWN_PROCESS gpu=${gpu} pid=${pid} free_mib=${free_mib}" \
          >>"${log_path}"
        kill -TERM "${pid}"
      fi
    done < <(
      nvidia-smi \
        -i "${gpu}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits
    )
  done
  sleep 30
done
