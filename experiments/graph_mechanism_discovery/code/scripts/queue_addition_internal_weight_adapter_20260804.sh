#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]] || (( ($# - 2) % 2 != 0 )); then
  echo "usage: $0 <physical_gpu> <wait_tmux_session> <baseline> <site> [<baseline> <site> ...]" >&2
  exit 2
fi

physical_gpu="$1"
wait_session="$2"
shift 2

while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep 20
done

launcher="/data/wujiaju/LooPlus/scripts/run_addition_internal_weight_adapter_20260804.sh"
log_root="/data/wujiaju/logs/paper_length_telomere_20260731/addition_internal_weight_adapter_20260804"

while (( $# )); do
  baseline="$1"
  site="$2"
  shift 2
  label="${baseline}_seed0_${site}"
  log_dir="${log_root}/${label}"
  mkdir -p "$log_dir"
  set +e
  bash "$launcher" "$baseline" "$site" "$physical_gpu" \
    > "${log_dir}/runner.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "${log_dir}/exit_code"
  if [[ $code -ne 0 ]]; then
    echo "queue stopped after failed job: $label" >&2
    exit "$code"
  fi
done
