set -eu
worker_pid="$1"
gpu="$2"
helper="$3"
while kill -0 "$worker_pid" 2>/dev/null; do sleep 10; done
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 100 ]; then echo "GPU no longer empty; helper not started"; exit 1; fi
CUDA_VISIBLE_DEVICES="$gpu" /data/paperexperiment/.venvs/loopreasoner/bin/python -u /data/paperexperiment/fig6_retest_20260924/parity_helper.py --out "/data/paperexperiment/fig6_retest_20260924/helper$helper" --shard 2 --shards 3 --helper
