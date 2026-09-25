#!/usr/bin/env bash
# Queue the small long-schedule evaluation until a shared A100 is genuinely
# safe.  This is deliberately conservative: two readings 60 seconds apart,
# low utilization, a 16 GiB reserve for existing jobs, and a 2 GiB allowance
# for this evaluation (the matching trainer peaked below 0.5 GiB).
set -euo pipefail

cd /data/wujiaju/LooPlus
export PYTHONPATH=/data/wujiaju/LooPlus

run_root=/data/wujiaju/graph_path_fixed_h1_j_20260801/long_j7_schedule_eval
mkdir -p "$run_root" /data/wujiaju/logs
monitor_log="$run_root/shared_gpu_monitor.log"
reserve_mib=$((16 * 1024))
new_job_allowance_mib=$((2 * 1024))
minimum_free_mib=$((reserve_mib + new_job_allowance_mib))
max_utilization=10

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

candidate_gpu() {
  gpu_snapshot | awk -F, -v minimum_free="$minimum_free_mib" -v max_util="$max_utilization" '
    {
      for (i = 1; i <= NF; ++i) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
      free_mib = $2 - $3
      if (free_mib >= minimum_free && $4 <= max_util) {
        print $1
        exit
      }
    }
  '
}

gpu_used_mib() {
  gpu_snapshot | awk -F, -v wanted="$1" '
    {
      for (i = 1; i <= NF; ++i) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
      if ($1 == wanted) { print $3; exit }
    }
  '
}

gpu_utilization() {
  gpu_snapshot | awk -F, -v wanted="$1" '
    {
      for (i = 1; i <= NF; ++i) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
      if ($1 == wanted) { print $4; exit }
    }
  '
}

echo "[$(date -Is)] waiting for a safe GPU; min_free=${minimum_free_mib}MiB max_util=${max_utilization}%" | tee -a "$monitor_log"
while true; do
  candidate="$(candidate_gpu || true)"
  if [[ -z "$candidate" ]]; then
    sleep 30
    continue
  fi
  used_before="$(gpu_used_mib "$candidate")"
  util_before="$(gpu_utilization "$candidate")"
  {
    echo "[$(date -Is)] candidate gpu=${candidate} used=${used_before}MiB util=${util_before}%"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
    if [[ -n "$pids" ]]; then
      ps -o pid=,user=,etime=,pcpu=,command= -p "$pids"
    fi
  } >> "$monitor_log"
  sleep 60
  used_after="$(gpu_used_mib "$candidate")"
  util_after="$(gpu_utilization "$candidate")"
  total_mib="$(gpu_snapshot | awk -F, -v wanted="$candidate" '
    { for (i = 1; i <= NF; ++i) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
      if ($1 == wanted) { print $2; exit } }')"
  free_after=$((total_mib - used_after))
  delta=$((used_after - used_before))
  if (( delta < 0 )); then delta=$(( -delta )); fi
  if (( free_after < minimum_free_mib || util_after > max_utilization || delta > 512 )); then
    echo "[$(date -Is)] reject gpu=${candidate} used_delta=${delta}MiB free=${free_after}MiB util=${util_after}%" >> "$monitor_log"
    sleep 30
    continue
  fi
  echo "[$(date -Is)] launch on gpu=${candidate}; stable_delta=${delta}MiB free=${free_after}MiB" | tee -a "$monitor_log"
  CUDA_VISIBLE_DEVICES="$candidate" bash scripts/run_graph_path_fixed_j7_long_schedules_remote_20260801.sh &
  job_pid=$!
  echo "[$(date -Is)] evaluation_pid=${job_pid}" | tee -a "$monitor_log"
  while kill -0 "$job_pid" 2>/dev/null; do
    sleep 30
    used_now="$(gpu_used_mib "$candidate")"
    total_now="$(gpu_snapshot | awk -F, -v wanted="$candidate" '
      { for (i = 1; i <= NF; ++i) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
        if ($1 == wanted) { print $2; exit } }')"
    free_now=$((total_now - used_now))
    echo "[$(date -Is)] running gpu=${candidate} used=${used_now}MiB free=${free_now}MiB" >> "$monitor_log"
    if (( free_now < reserve_mib )); then
      echo "[$(date -Is)] stopping only evaluation pid=${job_pid}: reserve violated" | tee -a "$monitor_log"
      kill "$job_pid" || true
      wait "$job_pid" || true
      exit 1
    fi
  done
  wait "$job_pid"
  echo "[$(date -Is)] evaluation completed" | tee -a "$monitor_log"
  exit 0
done
