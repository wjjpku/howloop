#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-6}"
python_bin=/data/wujiaju/.venvs/loopreasoner/bin/python
code=/data/wujiaju/LooPlus
root=/data/wujiaju/graph_path_telomere_canonical_diag_lora_20260731
log=/data/wujiaju/logs/graph_path_telomere_canonical_diag_lora_20260731/extended_circuit_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_mechanism_queue 2>/dev/null; do sleep 30; done

consecutive=0
while [[ "${consecutive}" -lt 2 ]]; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s gpu=%s free_mib=%s capacity_check=%s\n' "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${log}"
    if [[ "${free}" -ge 19456 ]]; then consecutive=$((consecutive + 1)); else consecutive=0; fi
    [[ "${consecutive}" -lt 2 ]] && sleep 60
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${code}"

"${python_bin}" -m reasoning_loop.graph_path_telomere_diag_rank48_circuit \
    --checkpoint /data/wujiaju/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
    --phase-summary "${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json" \
    --operator-artifact "${root}/controllers/final_seed0_ce_h64/task_lora_j.pt" \
    --operator-label task_diagonal_lora_r48_seed211001 \
    --out-dir "${root}/audits/final_seed0_ce_h64/extended_circuit" \
    --device cuda --batch-size 256 --cycles 1 32 64 --mode-cycles 1 32 64 \
    --skip-joint-hybrids --seed 214101 --cuda-memory-fraction 0.035 || true

printf '%s status=queue_complete\n' "$(date -Is)" >> "${log}"
