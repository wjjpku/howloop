#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-7}"
runner=/data/paperexperiment/LooPlus/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh
checkpoint=/data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=/data/paperexperiment/LooPlus/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
log=/data/paperexperiment/logs/graph_path_telomere_canonical_diag_lora_20260731/horizon_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_identity 2>/dev/null; do sleep 30; done

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

for horizon in 24 32 48 96; do
    wait_for_capacity
    label="final_seed0_rank48_ce_h${horizon}"
    printf '%s starting=%s\n' "$(date -Is)" "${label}" >> "${log}"
    SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 \
        bash "${runner}" "${gpu}" "${label}" "${checkpoint}" "${phase}" \
        "final-only CE at loop 8" 0 "${horizon}" 211001 || true
done
