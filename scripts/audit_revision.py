#!/usr/bin/env python3
"""Recompute revised-submission statistics from archived per-example records."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'experiments/revision'
OUT=ROOT/'outputs/audit';OUT.mkdir(parents=True,exist_ok=True)
summary=json.loads((REV/'composition_summary.json').read_text())['results']['A']
records=[np.load(REV/f'long_composition/A_fit{fit}.npz') for fit in (1,2)]
labels=records[0]['labels']
assert np.array_equal(labels,records[1]['labels'])
mask=np.all(np.diff(np.sort(labels[:,:5],axis=1),axis=1)!=0,axis=1)
assert int(mask.sum())==summary['n_distinct']==3200
long={}
for name,columns in [('one',[0]),('two',[1]),('mixed',list(range(2,34)))]:
 values=[];prefix=[]
 for step in range(8):
  accuracies=[];all_prefix=[]
  for d in records:
   cumulative=np.cumsum(d['sequences'],axis=1)
   correct=np.stack([d['predictions'][mask,:,j]==labels[mask][:,cumulative[:,j]] for j in range(step+1)],axis=-1)
   accuracies.append(float(correct[:,columns,-1].mean()))
   all_prefix.append(float(correct[:,columns].all(-1).mean()))
  values.append(float(np.mean(accuracies)*100))
  prefix.append(float(np.mean(all_prefix)*100))
 assert np.allclose(values,summary['groups'][name]['endpoint_pct'],atol=1e-9)
 assert np.allclose(prefix,summary['groups'][name]['prefix_pct'],atol=1e-9)
 long[name]={'endpoint_pct':values,'every_prefix_pct':prefix}
(OUT/'long_composition.json').write_text(json.dumps({'n':int(mask.sum()),'groups':long},indent=2)+'\n')
print('Eight-step controller reuse matches saved per-example predictions.')
