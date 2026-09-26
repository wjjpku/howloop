#!/usr/bin/env python3
"""Recompute the Qwen appendix table from archived complete-answer predictions."""
from collections import defaultdict
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT.parents[1] / 'provenance/additional_controls.json').read_text())['qwen']
assert hashlib.sha256((ROOT / 'results/predictions.jsonl').read_bytes()).hexdigest() == record['prediction_sha256']
rows = [json.loads(line) for line in (ROOT / 'results/predictions.jsonl').open()]
panel = json.loads((ROOT / 'results/panel.json').read_text())
saved = json.loads((ROOT / 'results/summary.json').read_text())
arms = {'normal', 'no_j_after4', 'no_f_after4', 'prefix4'}
assert len(panel) == 1280 and len(rows) == 5120
assert {row['arm'] for row in rows} == arms
by_arm = defaultdict(list)
for row in rows:
    reference = panel[row['panel_id']]
    assert 0 <= row['panel_id'] < 1280
    assert all(row[key] == reference[key] for key in ('task', 'steps', 'prompt', 'answer'))
    assert row['whole_correct'] == (row['generated'].strip().casefold() == str(row['answer']).strip().casefold())
    by_arm[row['arm']].append(row)

result = {}
for arm in sorted(arms):
    group = by_arm[arm]
    assert len(group) == 1280 and {row['panel_id'] for row in group} == set(range(1280))
    def stats(subset):
        return {'n': len(subset), 'whole': sum(row['whole_correct'] for row in subset)}
    short = [row for row in group if row['steps'] <= 4]
    long = [row for row in group if row['steps'] >= 5]
    result[arm] = {'all': stats(group), 'short': stats(short), 'long': stats(long),
                   'per_task_long': {task: stats([row for row in long if row['task'] == task])
                                     for task in sorted({row['task'] for row in long})}}
    for key in ('all', 'short', 'long'):
        assert result[arm][key]['n'] == saved[arm][key]['n']
        assert result[arm][key]['whole'] == saved[arm][key]['whole']
    for task, value in result[arm]['per_task_long'].items():
        assert value['n'] == saved[arm]['per_task_long'][task]['n']
        assert value['whole'] == saved[arm]['per_task_long'][task]['whole']

assert [result[a]['long']['whole'] for a in ('normal', 'no_j_after4', 'no_f_after4', 'prefix4')] == [574, 183, 200, 164]
print(json.dumps(result, indent=2))
