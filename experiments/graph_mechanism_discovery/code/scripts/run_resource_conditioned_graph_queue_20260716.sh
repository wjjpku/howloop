#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_INDEX SEED" >&2
  exit 2
fi

gpu_index=$1
seed=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_resource_conditioned_graph_20260716.sh" \
  "$gpu_index" compressor "$seed"
"$script_dir/run_resource_conditioned_graph_20260716.sh" \
  "$gpu_index" standard_transition "$seed"
