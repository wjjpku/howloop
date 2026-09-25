#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
code=/data/wujiaju/LooPlus
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
root=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731
run=final_seed0_rank96_ce_h64
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/${run}_audit_queue_gpu${gpu}.log

while tmux has-session -t '=telomere_canonical_rank96_h96_audit' 2>/dev/null; do
    sleep 30
done

consecutive=0
while [[ "${consecutive}" -lt 2 ]]; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s gpu=%s free_mib=%s capacity_check=%s\n' \
        "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${log}"
    if [[ "${free}" -ge 19456 ]]; then consecutive=$((consecutive + 1)); else consecutive=0; fi
    [[ "${consecutive}" -lt 2 ]] && sleep 60
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
cd "${code}"

"${python_bin}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
    --phase-summary "${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json" \
    --affine-artifact "${root}/initializers/${run}/unit_j_maps.pt" \
    --affine-label explicit_diag_rrr_r96 \
    --lora-artifacts "${root}/controllers/${run}/task_lora_j.pt" \
    --out-dir "${root}/audits/${run}/strict_unseen" \
    --device cuda --sample-per-partition 512 --batch-size 32 \
    --continuation-loops 128 --sample-seed 215101 \
    --cuda-memory-fraction 0.035 --training-graph-protocol canonical_artifact

"${python_bin}" scripts/analyze_diagonal_low_rank_spectrum.py \
    --artifact "${root}/controllers/${run}/task_lora_j.pt" \
    --label task_diagonal_lora_r96_seed211001 \
    --out-dir "${root}/audits/${run}/spectrum"

printf '%s status=queue_complete\n' "$(date -Is)" >> "${log}"
