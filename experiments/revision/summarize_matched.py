"""Report every completed map fit without turning a partial run into a cohort conclusion."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path('/data/paperexperiment/reviewer_revision_20260926/matched');OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
rows=[];missing=[]
for seed in range(5):
 for regime in ['final','stepwise']:
  d=ROOT/'runs'/f'{regime}_seed{seed}'
  for fit in [1,2]:
   for hop,target in enumerate(['stay','one','two']):
    f=d/'maps'/f'{target}_fit{fit}.npz';meta=f.with_suffix('.json')
    if not f.exists() or not meta.exists():missing.append(f'{regime}/{seed}/{target}/{fit}');continue
    z=np.load(f);m=z['distinct'];y=z['labels'][:,hop];raw=z['native'];pre=z['pre'];pred=z['pred'];a=json.loads(meta.read_text())
    assert len(y)==5120 and int(m.sum())==a['test_n_distinct'];assert abs((pred[m]==y[m]).mean()-a['test_accuracy_distinct'])<1e-6
    rows.append({'seed':seed,'regime':regime,'target':target,'fit':fit,'n_all':len(y),'n_distinct':int(m.sum()),'native':float((raw[m]==y[m]).mean()),'pre_F':float((pre[m]==y[m]).mean()),'full_J':float((pred[m]==y[m]).mean()),'gain':float((pred[m]==y[m]).mean()-(raw[m]==y[m]).mean()),'best_step':a['best_step'],'all_accuracy':float((pred==y).mean()),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
paired=[]
for seed in range(5):
 for target in ['stay','one','two']:
  rr=[r for r in rows if r['seed']==seed and r['target']==target]
  if len(rr)!=4:continue
  p={'seed':seed,'target':target}
  for regime in ['final','stepwise']:
   for metric in ['native','pre_F','full_J','gain']:p[f'{regime}_{metric}']=float(np.mean([r[metric] for r in rr if r['regime']==regime]))
  p['stepwise_minus_final']=p['stepwise_full_J']-p['final_full_J'];paired.append(p)
summary={'complete':len(rows)==60,'expected_fits':60,'completed_fits':len(rows),'missing':missing,'rows':rows,'paired_results':paired,'scope':'N10 fixed depth8 D8L8 matched-loss protocol; 5 paired backbone seeds, two map fits; not the original D8L6 cohort','inference':'Backbone is the unit of replication. Do not pool map fits as independent backbones. Report native baselines and pre-F alongside full J; a native target already at ceiling is not an elicited steering gain.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
if rows:
 with (OUT/'fits.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'complete':summary['complete'],'completed_fits':len(rows),'paired_targets':len(paired),'rows':[{k:r[k] for k in ['seed','regime','target','fit','native','pre_F','full_J']} for r in rows]},indent=2))
