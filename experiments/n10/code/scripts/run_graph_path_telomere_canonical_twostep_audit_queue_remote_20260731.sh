#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
root=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731
log=/data/paperexperiment/logs/graph_path_telomere_canonical_diag_lora_20260731/twostep_audit_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_extended_queue 2>/dev/null; do sleep 30; done

wait_for_capacity() {
    local consecutive=0
    while [[ "${consecutive}" -lt 2 ]]; do
        local free
        free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        printf '%s gpu=%s free_mib=%s capacity_check=%s\n' "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${log}"
        if [[ "${free}" -ge 19456 ]]; then consecutive=$((consecutive + 1)); else consecutive=0; fi
        [[ "${consecutive}" -lt 2 ]] && sleep 60
    done
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

run_audit() {
    local run="$1" checkpoint="$2" phase="$3" seed="$4"
    wait_for_capacity
    "${python_bin}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
        --checkpoint "${checkpoint}" --phase-summary "${phase}" \
        --affine-artifact "${root}/initializers/${run}/unit_j_maps.pt" \
        --affine-label explicit_diag_rrr_r48 \
        --lora-artifacts "${root}/controllers/${run}/task_lora_j.pt" \
        --out-dir "${root}/audits/${run}/strict_unseen" \
        --device cuda --sample-per-partition 256 --batch-size 32 \
        --continuation-loops 128 --sample-seed "${seed}" \
        --cuda-memory-fraction 0.035 --training-graph-protocol canonical_artifact || true
}

run_audit final_seed1_ce_h64 \
    /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt \
    "${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed1.json" 214201

run_audit component_seed1_ce_h64 \
    /data/paperexperiment/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt \
    "${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_component_seed1.json" 214301

printf '%s status=queue_complete\n' "$(date -Is)" >> "${log}"
