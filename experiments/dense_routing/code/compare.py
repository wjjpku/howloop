from pathlib import Path
import json,numpy as np,gzip
from collections import defaultdict
P=Path(__file__).resolve().parents[1];OLD=Path('/data/wujiaju/paper_strengthening_20260925');B=Path('/data/wujiaju/n10_migration_20260923')
summary={};rng=np.random.default_rng(2026092601);w=rng.multinomial(512,np.full(512,1/512),size=5000)
def stat(hits,mask):
 h=np.mean(hits,axis=0)*mask;n=mask.reshape(512,10).sum(1);g=h.reshape(512,10).sum(1);den=w@n;v=(w@g)/np.maximum(den,1);return dict(mean=float(h.sum()/mask.sum()) if mask.any() else None,ci95=np.quantile(v[den>0],[.025,.975]).tolist() if mask.any() else None,n=int(mask.sum()),fits=[float((x*mask).sum()/mask.sum()) if mask.any() else None for x in hits])
for name in ['A','C','B','D','E']:
 if not (P/f'composition/{name}_fit2.json').exists():continue
 ds=[np.load(P/f'graph/{name}_fit{f}.npz') for f in [1,2]];old=[np.load(OLD/f'graph/{name}_fit{f}.npz') for f in [1,2]];l=ds[0]['labels'];mask=np.all(np.diff(np.sort(l,axis=1),axis=1)!=0,axis=1);row={'single':{},'matrix':{},'composition':{},'pre_executor':{}}
 for j,k in enumerate(['raw','one','two']):
  target={'raw':1,'one':1,'two':2}[k];a=[d['native'][:,j]==l[:,target] for d in ds];b=[d['native'][:,j]==l[:,target] for d in old];row['single'][k]={'dense':stat(a,mask),'lowrank':stat(b,mask),'delta':stat([x.astype(float)-y for x,y in zip(a,b)],mask)}
 for j,k in [(1,'one'),(2,'two')]:
  row['pre_executor'][k]={'dense':stat([d['pre'][:,j]==l[:,j] for d in ds],mask),'lowrank':stat([d['pre'][:,j]==l[:,j] for d in old],mask)}
 for i,cond in enumerate(ds[0]['conditions']):
  source=str(cond).split('_to_')[0];target={'raw':0,'one':1,'two':2}[source]
  a=[d['predictions'][i]==l[:,target] for d in ds];b=[d['predictions'][i]==l[:,target] for d in old];row['matrix'][str(cond)]={'dense':stat(a,mask),'lowrank':stat(b,mask),'delta':stat([x.astype(float)-y for x,y in zip(a,b)],mask)}
 cs=[np.load(P/f'composition/{name}_fit{f}.npz') for f in [1,2]];co=[np.load(Path('/data/wujiaju/continuous_composition_20260925')/f'{name}_fit{f}.npz') for f in [1,2]];l=cs[0]['labels'];mask=np.all(np.diff(np.sort(l,axis=1),axis=1)!=0,axis=1)
 for seq,target in [('one_one',2),('one_two',3),('two_one',3),('two_two',4)]:
  i=list(cs[0]['sequences']).index(seq);a=[d['second'][:,i]==l[:,target] for d in cs];b=[d['second'][:,i]==l[:,target] for d in co];row['composition'][seq]={'dense':stat(a,mask),'lowrank':stat(b,mask),'delta':stat([x.astype(float)-y for x,y in zip(a,b)],mask)}
 summary[name]=row
(P/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:{'single':{n:v['dense']['mean'] for n,v in s['single'].items()},'composition':{n:v['dense']['mean'] for n,v in s['composition'].items()}} for k,s in summary.items()},indent=2))
