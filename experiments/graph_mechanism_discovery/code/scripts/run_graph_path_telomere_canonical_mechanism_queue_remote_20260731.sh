#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-6}"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
runner=${code}/scripts/run_graph_path_telomere_canonical_diag_lora_backbone_remote_20260731.sh
root=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731
checkpoint=/data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt
phase=${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json
artifact=${root}/controllers/final_seed0_ce_h64/task_lora_j.pt
label=task_diagonal_lora_r48_seed211001
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/mechanism_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_backbone_queue 2>/dev/null; do sleep 30; done

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

wait_for_capacity
PLACEMENT=pre_block2 SHARED_GPU=1 DECLARED_PEAK_GIB=3 CUDA_MEMORY_FRACTION=0.035 \
    bash "${runner}" "${gpu}" final_seed0_preblock2_rank48_ce_h64 \
    "${checkpoint}" "${phase}" "final-only CE at loop 8" 0 64 \
    211001 311001 411001 || true

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${code}"

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint "${checkpoint}" \
    --phase-summary "${phase}" \
    --affine-artifact "${root}/initializers/final_seed0_preblock2_rank48_ce_h64/unit_j_maps.pt" \
    --affine-label explicit_diag_rrr_r48 \
    --lora-artifacts "${root}/controllers/final_seed0_preblock2_rank48_ce_h64/task_lora_j.pt" \
    --out-dir "${root}/audits/final_seed0_preblock2_rank48_ce_h64/strict_unseen" \
    --device cuda --sample-per-partition 512 --batch-size 32 \
    --continuation-loops 128 --sample-seed 213701 \
    --cuda-memory-fraction 0.035 --training-graph-protocol canonical_artifact || true

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_diagonal_low_rank_ablation \
    --checkpoint "${checkpoint}" --phase-summary "${phase}" \
    --artifact "${artifact}" --label "${label}" \
    --out-dir "${root}/audits/final_seed0_ce_h64/diag_ablation" \
    --device cuda --sample-per-partition 512 --batch-size 32 \
    --continuation-loops 64 --sample-seed 213801 --shuffle-seed 213802 \
    --cuda-memory-fraction 0.035 || true

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_diagonal_low_rank_impulse \
    --checkpoint "${checkpoint}" --phase-summary "${phase}" \
    --artifact "${artifact}" --label "${label}" \
    --out-dir "${root}/audits/final_seed0_ce_h64/diag_impulse" \
    --device cuda --damage-cycles 1 32 64 --batch-size 32 --batches 16 \
    --continuation-loops 64 --seed 213901 --cuda-memory-fraction 0.035 || true

wait_for_capacity
"${python_bin}" -m reasoning_loop.graph_path_telomere_boundary_norm_control \
    --checkpoint "${checkpoint}" --phase-summary "${phase}" \
    --artifact "${artifact}" --label "${label}" \
    --out-dir "${root}/audits/final_seed0_ce_h64/norm_control" \
    --device cuda --sample-per-partition 128 --batch-size 16 \
    --continuation-loops 128 --sample-seed 214001 \
    --cuda-memory-fraction 0.035 || true

printf '%s status=queue_complete\n' "$(date -Is)" >> "${log}"
