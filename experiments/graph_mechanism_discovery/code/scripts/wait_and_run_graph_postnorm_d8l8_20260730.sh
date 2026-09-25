#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data/wujiaju/LooPlus_postnorm_20260730"
python_bin="/data/wujiaju/.venvs/loopreasoner/bin/python"
experiment_root="/data/wujiaju/graph_path_postnorm_D8L8_20260730"
log_root="/data/wujiaju/logs/graph_path_postnorm_D8L8_20260730"
orchestrator_log="${log_root}/orchestrator.log"
status_path="${experiment_root}/orchestrator_status.txt"

mkdir -p "${experiment_root}" "${log_root}"
cd "${repo_dir}"

record_status() {
  local message="$1"
  echo "$(date --iso-8601=seconds) ${message}" | tee -a "${orchestrator_log}"
  echo "$(date --iso-8601=seconds) ${message}" >"${status_path}"
}

idle_gpu() {
  local gpu
  for gpu in $(seq 0 7); do
    if gpu_is_still_idle "${gpu}"; then
      echo "${gpu}"
      return
    fi
  done
}

gpu_is_still_idle() {
  local gpu="$1"
  local row
  local compute_pids
  row="$(
    nvidia-smi \
      -i "${gpu}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )"
  compute_pids="$(
    nvidia-smi \
      -i "${gpu}" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits
  )"
  [[ -z "${compute_pids}" ]] &&
    awk -F',' '$1 + 0 < 1000 && $2 + 0 < 5 {exit 0} {exit 1}' <<<"${row}"
}

record_status "WAITING_FOR_EMPTY_GPU"
selected_gpu=""
while [[ -z "${selected_gpu}" ]]; do
  candidate="$(idle_gpu)"
  if [[ -z "${candidate}" ]]; then
    sleep 30
    continue
  fi
  record_status "IDLE_CANDIDATE gpu=${candidate}; rechecking_after_60s"
  sleep 60
  if gpu_is_still_idle "${candidate}"; then
    selected_gpu="${candidate}"
  else
    record_status "CANDIDATE_LOST gpu=${candidate}"
  fi
done

smoke_root="${experiment_root}/a100_smoke_$(date +%Y%m%dT%H%M%S)"
smoke_log="${log_root}/a100_smoke.log"
record_status "SMOKE_START gpu=${selected_gpu} out=${smoke_root}"
CUDA_VISIBLE_DEVICES="${selected_gpu}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${python_bin}" -u -m reasoning_loop.graph_path_loop \
  --node-count 8 \
  --max-depth 8 \
  --d-model 256 \
  --n-heads 4 \
  --d-mlp 1024 \
  --n-layers 2 \
  --loops 8 \
  --steps 20 \
  --batch-size 512 \
  --eval-batch-size 1024 \
  --eval-batches 1 \
  --eval-every 20 \
  --print-every 20 \
  --lr 0.0003 \
  --weight-decay 0.3 \
  --warmup-steps 20 \
  --grad-clip 1.0 \
  --seed 9000 \
  --dropout 0.0 \
  --inner-norm-style post_layernorm \
  --aux-loss 0.0 \
  --device cuda \
  --amp \
  --no-compile \
  --no-save-checkpoints \
  --out-dir "${smoke_root}" \
  >"${smoke_log}" 2>&1

smoke_summary="$(
  find "${smoke_root}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit
)"
test -s "${smoke_summary}"
peak_reserved="$(
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["peak_cuda_memory_reserved_mib"])' \
    "${smoke_summary}"
)"
record_status \
  "SMOKE_COMPLETE gpu=${selected_gpu} peak_reserved_mib=${peak_reserved}"

free_memory="$(
  nvidia-smi \
    -i "${selected_gpu}" \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits |
    tr -d ' '
)"
required_memory="$(
  "${python_bin}" -c \
    'import math,sys; print(math.ceil(float(sys.argv[1]) + 16384))' \
    "${peak_reserved}"
)"
if (( free_memory < required_memory )); then
  record_status \
    "ABORT_INSUFFICIENT_RESERVE gpu=${selected_gpu} free_mib=${free_memory} required_mib=${required_memory}"
  exit 1
fi

record_status \
  "FORMAL_START gpu=${selected_gpu} seeds=0,1,2,3,4,5 free_mib=${free_memory}"
setsid bash "${repo_dir}/scripts/train_graph_postnorm_d8l8_seed_20260730.sh" \
  "${selected_gpu}" 0 1 2 3 4 5 \
  >>"${orchestrator_log}" 2>&1 &
formal_pid="$!"

while kill -0 "${formal_pid}" 2>/dev/null; do
  free_memory="$(
    nvidia-smi \
      -i "${selected_gpu}" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits |
      tr -d ' '
  )"
  if (( free_memory < 16384 )); then
    record_status \
      "STOP_OWN_JOB_LOW_RESERVE gpu=${selected_gpu} free_mib=${free_memory} pid=${formal_pid}"
    kill -TERM -- "-${formal_pid}"
    wait "${formal_pid}" || true
    exit 1
  fi
  sleep 30
done

wait "${formal_pid}"
for seed in 0 1 2 3 4 5; do
  checkpoint="${experiment_root}/training/D8_L8_postnorm_seed${seed}/graphpath_N8_D8_d256_B2_L8_seed${seed}/final.pt"
  test -s "${checkpoint}"
done
"${python_bin}" "${repo_dir}/scripts/summarize_graph_postnorm_training_20260730.py" \
  --post-root "${experiment_root}/training" \
  --pre-root "/data/wujiaju/graph_path_compression_circuit_20260725/training" \
  --out-dir "${experiment_root}/training_comparison"
record_status \
  "FORMAL_COMPLETE gpu=${selected_gpu} seeds=0,1,2,3,4,5 summary=${experiment_root}/training_comparison/training_summary.json"
