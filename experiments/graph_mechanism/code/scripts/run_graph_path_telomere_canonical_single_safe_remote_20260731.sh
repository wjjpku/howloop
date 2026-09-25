#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 9 ]]; then
    echo "usage: $0 GPU RUN CHECKPOINT PHASE LOSS STATE_WEIGHT HORIZON RANK SEED [SEED ...]" >&2
    exit 2
fi

gpu="$1"
run="$2"
checkpoint="$3"
phase="$4"
loss="$5"
state_weight="$6"
horizon="$7"
rank="$8"
shift 8
seeds=("$@")
runner=/data/wujiaju/LooPlus/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/${run}_safe_queue_gpu${gpu}.log

consecutive=0
while [[ "${consecutive}" -lt 2 ]]; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s gpu=%s free_mib=%s capacity_check=%s\n' \
        "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${log}"
    if [[ "${free}" -ge 19456 ]]; then
        consecutive=$((consecutive + 1))
    else
        consecutive=0
    fi
    [[ "${consecutive}" -lt 2 ]] && sleep 60
done

SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 \
    RANK="${rank}" PLACEMENT=loop_boundary \
    bash "${runner}" "${gpu}" "${run}" "${checkpoint}" "${phase}" \
    "${loss}" "${state_weight}" "${horizon}" "${seeds[@]}"
