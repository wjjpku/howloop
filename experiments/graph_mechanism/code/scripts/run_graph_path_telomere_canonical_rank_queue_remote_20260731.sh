#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-1}"
runner=/data/wujiaju/LooPlus/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/wujiaju/LooPlus/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/rank_queue_gpu${gpu}.log

while tmux has-session -t '=telomere_canonical_loss' 2>/dev/null; do sleep 30; done

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

for rank in 8 16 32 64 96; do
    wait_for_capacity
    label="final_seed0_rank${rank}_ce_h64"
    printf '%s starting=%s\n' "$(date -Is)" "${label}" >> "${log}"
    SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 RANK="${rank}" \
        bash "${runner}" "${gpu}" "${label}" "${checkpoint}" "${phase}" \
        "final-only CE at loop 8" 0 64 211001 || true
done
