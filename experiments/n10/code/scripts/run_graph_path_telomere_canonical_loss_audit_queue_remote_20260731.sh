#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-6}"
python_bin=/data/paperexperiment/.venvs/loopreasoner/bin/python
code=/data/paperexperiment/LooPlus
root=/data/paperexperiment/graph_path_telomere_canonical_diag_lora_20260731
log=/data/paperexperiment/logs/graph_path_telomere_canonical_diag_lora_20260731/loss_audit_queue_gpu${gpu}.log

while tmux has-session -t telomere_canonical_twostep_queue 2>/dev/null; do sleep 30; done
consecutive=0
while [[ "${consecutive}" -lt 2 ]]; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s gpu=%s free_mib=%s capacity_check=%s\n' "$(date -Is)" "${gpu}" "${free}" "${consecutive}" >> "${log}"
    if [[ "${free}" -ge 19456 ]]; then consecutive=$((consecutive + 1)); else consecutive=0; fi
    [[ "${consecutive}" -lt 2 ]] && sleep 60
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${code}"
cd "${code}"
"${python_bin}" -m reasoning_loop.graph_path_telomere_task_mlp_audit \
    --checkpoint /data/paperexperiment/graph_path_compression_circuit_20260725/training/D8_L8_seed0/graphpath_N8_D8_d256_B2_L8_seed0/best.pt \
    --phase-summary "${code}/results/graph_path_telomere_canonical_diag_lora_20260731/config/phase_final_seed0.json" \
    --affine-artifact "${root}/initializers/final_seed0_ceplusmse0p1_h64/unit_j_maps.pt" \
    --affine-label explicit_diag_rrr_r48 \
    --lora-artifacts "${root}/controllers/final_seed0_ceplusmse0p1_h64/task_lora_j.pt" \
    --out-dir "${root}/audits/final_seed0_ceplusmse0p1_h64/strict_unseen" \
    --device cuda --sample-per-partition 512 --batch-size 32 \
    --continuation-loops 128 --sample-seed 214401 \
    --cuda-memory-fraction 0.035 --training-graph-protocol canonical_artifact || true
printf '%s status=queue_complete\n' "$(date -Is)" >> "${log}"
