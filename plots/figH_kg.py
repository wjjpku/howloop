#!/usr/bin/env python3
"""Draw the KG length curve from the archived paired example predictions."""
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('PAPEREXPERIMENT_OUTPUT', ROOT/'outputs'))/'figures'
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in (ROOT/'experiments/kg/results/key_cells_1024.jsonl').open()]
by_length = {n: [] for n in range(1, 33)}
for row in rows:
    by_length[row['test_length']].append(row)
xs = list(range(1, 33))
raw = [100*sum(r['raw_prediction'] == r['target'] for r in by_length[n])/1024 for n in xs]
controlled = [100*sum(r['after_prediction'] == r['target'] for r in by_length[n])/1024 for n in xs]
fig, ax = plt.subplots(figsize=(6.36, 2.21))
ax.axvspan(.5, 3.5, color='#e3e6e9', alpha=.8)
ax.axvspan(3.5, 16.5, color='#e3edf8', alpha=.9)
ax.plot(xs, controlled, marker='o', markersize=2.4, linewidth=1.6, color='#2262a5', label='Frozen F with affine J')
ax.plot(xs, raw, marker='s', markersize=2.4, linewidth=1.6, color='#bd5933', label='Frozen F only')
ax.axhline(100/64, color='#888', linestyle=':', linewidth=1, label='Chance (1/64)')
ax.axvline(16.5, color='#888', linestyle='--', linewidth=1)
ax.set(xlim=(.5, 32.5), ylim=(-3, 110), xlabel='Relation-composition length n', ylabel='Target accuracy (%)')
ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', alpha=.18)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=7)
fig.subplots_adjust(left=.105, right=.99, bottom=.24, top=.79)
fig.savefig(OUT/'figH_kg_state_control.pdf')
plt.close(fig)
