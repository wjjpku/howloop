#!/usr/bin/env bash
set -euo pipefail

SESSION_LABEL=fixed_h1_r48_s16_product
QUEUE_LOG=/data/wujiaju/logs/graph_path_fixed_h1_j_20260801/queue.log
MANIFEST=/data/wujiaju/graph_path_fixed_h1_j_20260801/paired_fullbank_init/run_manifest.txt
RUNNER=/data/wujiaju/LooPlus/scripts/run_graph_path_fixed_h1_r48_s16_remote_20260801.sh
CANDIDATES=(3 4 5 6)
MIN_FREE_MIB=16852
MAX_UTIL=10
STABLE_FREE_DELTA_MIB=1024

mkdir -p "$(dirname "$QUEUE_LOG")" "$(dirname "$MANIFEST")"
exec >>"$QUEUE_LOG" 2>&1
echo "$(date -Is) queue_started label=$SESSION_LABEL candidates=${CANDIDATES[*]}"

sample_gpu() {
  local gpu=$1
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' -v target="$gpu" '$1 == target {print $2, $3, $4}'
}

selected=""
while [[ -z "$selected" ]]; do
  for gpu in "${CANDIDATES[@]}"; do
    read -r used1 free1 util1 < <(sample_gpu "$gpu")
    echo "$(date -Is) gate1 gpu=$gpu used=$used1 free=$free1 util=$util1"
    if (( free1 < MIN_FREE_MIB || util1 > MAX_UTIL )); then
      continue
    fi
    sleep 60
    read -r used2 free2 util2 < <(sample_gpu "$gpu")
    delta=$(( free2 - free1 ))
    if (( delta < 0 )); then
      delta=$(( -delta ))
    fi
    echo "$(date -Is) gate2 gpu=$gpu used=$used2 free=$free2 util=$util2 delta_free=$delta"
    if (( free2 >= MIN_FREE_MIB && util2 <= MAX_UTIL && delta <= STABLE_FREE_DELTA_MIB )); then
      selected=$gpu
      break
    fi
  done
  [[ -n "$selected" ]] || sleep 30
done

{
  echo "launch_time=$(date -Is)"
  echo "physical_gpu=$selected"
  echo "declared_peak_mib=468"
  echo "reserve_mib=16384"
  echo "minimum_free_mib=$MIN_FREE_MIB"
  echo "shared_card=true"
  echo "preexisting_compute_processes:"
  selected_uuid=$(nvidia-smi -i "$selected" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader,nounits | awk -F', ' -v uuid="$selected_uuid" '$1 == uuid'
  echo "preexisting_process_owners:"
  selected_pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid \
    --format=csv,noheader,nounits | awk -F', ' -v uuid="$selected_uuid" '$1 == uuid {print $2}' | paste -sd, -)
  if [[ -n "$selected_pids" ]]; then
    ps -o user:16,pid,lstart,cmd -p "$selected_pids"
  fi
} >"$MANIFEST"

echo "$(date -Is) launching physical_gpu=$selected"
setsid env CUDA_VISIBLE_DEVICES="$selected" "$RUNNER" &
job_pid=$!
echo "job_process_group=$job_pid" >>"$MANIFEST"

for _ in $(seq 1 10); do
  sleep 30
  if ! kill -0 "$job_pid" 2>/dev/null; then
    break
  fi
  read -r used free util < <(sample_gpu "$selected")
  echo "$(date -Is) startup_monitor gpu=$selected used=$used free=$free util=$util"
  if (( free < 16384 )); then
    echo "$(date -Is) reserve_violation stopping_own_process_group=$job_pid"
    kill -TERM -- "-$job_pid"
    wait "$job_pid" || true
    echo "status=stopped_reserve_violation" >>"$MANIFEST"
    exit 1
  fi
done

set +e
wait "$job_pid"
status=$?
set -e
echo "status=$status" >>"$MANIFEST"
echo "completion_time=$(date -Is)" >>"$MANIFEST"
echo "$(date -Is) finished status=$status physical_gpu=$selected"
