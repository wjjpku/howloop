#!/usr/bin/env python3
"""Check current-submission graph control and target-switching claims."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/audit';OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'experiments/composition'
a=[np.load(BASE/f'A_fit{fit}.npz') for fit in (1,2)]
labels=a[0]['labels'];assert np.array_equal(labels,a[1]['labels'])
mask=np.all(np.diff(np.sort(labels,axis=1),axis=1)!=0,axis=1)
assert int(mask.sum())==3200
results={}
for sequence,target,expected in [('one_one',2,93.64),('one_two',3,65.28),('two_one',3,74.50),('two_two',4,68.12)]:
 rates=[]
 for d in a:
  i=list(d['sequences']).index(sequence)
  rates.append(float((d['second'][mask,i]==labels[mask,target]).mean()*100))
 mean=float(np.mean(rates));assert abs(mean-expected)<.006
 results[sequence]={'fits_pct':rates,'mean_pct':mean}
selected={}
for seed in (10,13):
 summary=json.loads((ROOT/f'experiments/selected_mechanism/runs/seed{seed}/mechanism_summary.json').read_text())
 selected[str(seed)]={}
 for fit in (1,2):
  d=np.load(ROOT/f'experiments/selected_mechanism/exchange/seed{seed}_fit{fit}.npz')
  ids=d['labels'];valid=(ids[:,0]!=ids[:,1])&(ids[:,0]!=ids[:,2])&(ids[:,1]!=ids[:,2])
  assert int(valid.sum())==4149
  for row in (x for x in summary['target_exchange'] if x['fit']==fit):
   name=row['condition']
   if name.startswith('native_'):
    prediction=d['native'][valid,['raw','one','two'].index(name.removeprefix('native_'))]
   else:
    prediction=d['predictions'][list(d['conditions']).index(name),valid]
   for index,key in enumerate(('current','one','two')):
    actual=float((prediction==ids[valid,index]).mean())
    assert np.isclose(actual,row[key],rtol=0,atol=1e-12),(seed,fit,name,key)
  selected[str(seed)][str(fit)]={'n':int(valid.sum()),'conditions':len(summary['target_exchange'])//2}
(OUT/'submission_graph.json').write_text(json.dumps({'two_step_composition':results,'selected_target_exchange':selected},indent=2)+'\n')
print('Current-submission two-step composition and selected target exchange match saved predictions.')
