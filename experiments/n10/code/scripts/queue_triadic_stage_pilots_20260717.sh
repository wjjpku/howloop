#!/usr/bin/env bash
set -euo pipefail

repo_root=${RESOURCE_REUSE_REPO_ROOT:-/data/paperexperiment/LooPlus}
widths=${STAGE_COMPOSITION_WIDTHS:-"4 8 12 16 24"}

queue_one() {
  local gpu=$1
  local arm=$2
  local wait_session="gdepth_extension_gpu${gpu}"
  local session="stage_virtual_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already exists"
    return
  fi
  tmux new-session -d -s "$session" \
    "while tmux has-session -t '$wait_session' 2>/dev/null; do sleep 30; done; cd '$repo_root'; scripts/run_triadic_stage_composition_20260717.sh '$gpu' '$arm' 0 $widths"
  echo "queued $arm in $session after $wait_session"
}

queue_one 0 l1
queue_one 1 shared2
queue_one 2 unshared2
