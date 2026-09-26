#!/bin/sh
set -eu
idx=$1
root=/data/paperexperiment/ouro_all256_20260926
py=/data/paperexperiment/.venvs/loopreasoner/bin/python
"$py" -u "$root/code/parallel_semantics.py" --pairs "$root/semantic128_${idx}.json" --config "$root/config.json" --out "$root/semantic${idx}"
"$py" -u "$root/code/parallel_restore.py" --pairs "$root/restore128_${idx}.json" --out "$root/restore${idx}"
