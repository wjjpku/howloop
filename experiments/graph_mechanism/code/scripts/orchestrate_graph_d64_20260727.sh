#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
root="/data/wujiaju/graph_path_induction_contrast_d64_20260727"
repo="/data/wujiaju/LooPlus"
manifest="${root}/orchestration_manifest.txt"
log_root="/data/wujiaju/logs/graph_path_induction_contrast_d64_20260727"
mkdir -p "${root}" "${log_root}"

write_status() {
  local status="$1"
  printf 'status=%s\npid=%s\nphysical_gpu=%s\nupdated=%s\n' \
    "${status}" "$$" "${gpu}" "$(date --iso-8601=seconds)" >"${manifest}"
}

trap 'write_status failed' ERR
write_status running

wait_for_training() {
  while true; do
    local complete=1
    for seed in 0 1 2 3; do
      local path="${root}/training_manifest_seed${seed}.txt"
      if [[ ! -s "${path}" ]] || ! grep -q '^status=complete$' "${path}"; then
        complete=0
        break
      fi
    done
    if [[ "${complete}" -eq 1 ]]; then
      return
    fi
    sleep 30
  done
}

cd "${repo}"
echo "START d64 paired training $(date --iso-8601=seconds)"
for seed in 0 1 2 3; do
  session="graph_d64_seed${seed}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" \
      "cd ${repo} && bash scripts/train_graph_d64_paired_20260727.sh ${gpu} ${seed}"
  fi
done
wait_for_training
echo "COMPLETE d64 paired training $(date --iso-8601=seconds)"

echo "START d64 causal analyses $(date --iso-8601=seconds)"
bash scripts/run_graph_d64_paired_analyses_20260727.sh "${gpu}"
echo "COMPLETE d64 causal analyses $(date --iso-8601=seconds)"

write_status complete
echo "ALL COMPLETE $(date --iso-8601=seconds)"
