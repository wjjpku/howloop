#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
    echo "usage: $0 GPU RUN CHECKPOINT PHASE_JSON BACKBONE_LOSS AUDIT_SEED ORACLE_SEED" >&2
    exit 2
fi

gpu="$1"
run="$2"
checkpoint="$3"
phase="$4"
backbone_loss="$5"
audit_seed="$6"
oracle_seed="$7"
code=/data/wujiaju/LooPlus
root=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
log_root=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731
queue_log="${log_root}/${run}_queue_gpu${gpu}.log"

mkdir -p "${log_root}"
exec >> "${queue_log}" 2>&1

wait_for_capacity() {
    local consecutive=0
    while [[ "${consecutive}" -lt 2 ]]; do
        local free
        free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        printf '%s gpu=%s free_mib=%s capacity_check=%s\n' \
            "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${queue_log}"
        if [[ "${free}" -ge 19456 ]]; then
            consecutive=$((consecutive + 1))
        else
            consecutive=0
        fi
        [[ "${consecutive}" -lt 2 ]] && sleep 60
    done
    return 0
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

wait_for_capacity
SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 \
    RANK=48 PLACEMENT=loop_boundary \
    bash "${code}/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh" \
    "${gpu}" "${run}" "${checkpoint}" "${phase}" "${backbone_loss}" 0 64 \
    211001 311001 411001

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint "${checkpoint}" --phase-summary "${phase}" \
    --affine-artifact "${root}/initializers/${run}/unit_j_maps.pt" \
    --affine-label explicit_diag_rrr_r48 \
    --lora-artifacts "${root}/controllers/${run}/task_lora_j.pt" \
    --out-dir "${root}/audits/${run}/strict_unseen" \
    --device cuda --sample-per-partition 256 --batch-size 32 \
    --continuation-loops 128 --sample-seed "${audit_seed}" \
    --cuda-memory-fraction 0.035 --training-graph-protocol canonical_artifact

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_boundary_norm_control \
    --checkpoint "${checkpoint}" --phase-summary "${phase}" \
    --artifact "${root}/controllers/${run}/task_lora_j.pt" \
    --label task_diagonal_lora_r48_seed211001 \
    --out-dir "${root}/audits/${run}/boundary_oracle_controls" \
    --device cuda --sample-per-partition 64 --batch-size 16 \
    --continuation-loops 64 --sample-seed "${oracle_seed}" \
    --cuda-memory-fraction 0.035

printf '%s status=queue_complete run=%s gpu=%s\n' \
    "$(date -Is)" "${run}" "${gpu}" >> "${queue_log}"
