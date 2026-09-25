#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-6}"
runner=/data/wujiaju/LooPlus/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh
config_root=/data/wujiaju/LooPlus/results/graph_path_telomere_canonical_diag_lora_20260731/config
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/backbone_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_seed3_audit 2>/dev/null; do
    printf '%s waiting_for=telomere_canonical_seed3_audit\n' "$(date -Is)" >> "${log}"
    sleep 30
done

run_one() {
    local label="$1"
    local checkpoint="$2"
    local phase="$3"
    local backbone_loss="$4"
    printf '%s starting=%s gpu=%s\n' "$(date -Is)" "${label}" "${gpu}" >> "${log}"
    SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 \
        bash "${runner}" "${gpu}" "${label}" "${checkpoint}" "${phase}" \
        "${backbone_loss}" 0 64 211001 311001 411001
    local status=$?
    printf '%s finished=%s exit=%s\n' "$(date -Is)" "${label}" "${status}" >> "${log}"
    return "${status}"
}

run_one \
    final_seed1_ce_h64 \
    /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt \
    "${config_root}/phase_final_seed1.json" \
    "final-only CE at loop 8" || true

run_one \
    component_seed1_ce_h64 \
    /data/wujiaju/graph_path_prenorm_component_D8L8_20260731/training/full/D8_L8_full_seed1/graphpath_N8_D8_d256_B2_L8_seed1/best.pt \
    "${config_root}/phase_component_seed1.json" \
    "final CE at loop 8 plus intermediate CE on p_min(2t,D), t=1..7" || true
