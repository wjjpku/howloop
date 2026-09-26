#!/usr/bin/env python3
"""Recompute the KG appendix table and length curve from archived paired queries."""
from collections import defaultdict
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT.parents[1] / 'provenance/additional_controls.json').read_text())['kg']
assert hashlib.sha256((ROOT / 'results/key_cells_1024.jsonl').read_bytes()).hexdigest() == record['per_example_sha256']
assert hashlib.sha256((ROOT / 'results/key_cells_1024_summary.json').read_bytes()).hexdigest() == record['summary_sha256']
rows = [json.loads(line) for line in (ROOT / 'results/key_cells_1024.jsonl').open()]
saved = json.loads((ROOT / 'results/key_cells_1024_summary.json').read_text())
assert len(rows) == 32 * 1024
by_length = defaultdict(list)
for row in rows:
    length = row['test_length']
    assert 1 <= length <= 32 and len(row['relations']) == length
    by_length[length].append(row)

curve = []
for length in range(1, 33):
    group = by_length[length]
    assert len(group) == 1024 and {row['example'] for row in group} == set(range(1024))
    raw = sum(row['raw_prediction'] == row['target'] for row in group)
    before = sum(row['before_prediction'] == row['target'] for row in group)
    after = sum(row['after_prediction'] == row['target'] for row in group)
    cell = saved['cells'][length-1]
    assert (cell['test_length'], cell['n'], cell['raw_hits'], cell['before_hits'], cell['after_hits']) == (length, 1024, raw, before, after)
    curve.append({'length': length, 'n': 1024, 'raw_hits': raw, 'after_hits': after})

ranges = [(1, 3), (4, 16), (17, 24), (25, 32)]
table = []
for lo, hi in ranges:
    cells = curve[lo-1:hi]
    n = sum(x['n'] for x in cells)
    table.append({'lengths': f'{lo}-{hi}', 'n': n,
                  'raw_pct': round(100*sum(x['raw_hits'] for x in cells)/n, 2),
                  'with_j_pct': round(100*sum(x['after_hits'] for x in cells)/n, 2)})
assert [(x['n'], x['raw_pct'], x['with_j_pct']) for x in table] == [
    (3072, 99.87, 100.00), (13312, 2.53, 99.92),
    (8192, 1.54, 42.87), (8192, 1.56, 1.65)]
print(json.dumps({'table': table, 'lengths': len(curve)}, indent=2))
