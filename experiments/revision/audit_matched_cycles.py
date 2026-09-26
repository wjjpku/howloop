from pathlib import Path
import numpy as np,json,hashlib
R=Path('/data/paperexperiment/reviewer_revision_20260926/matched');out=[]
def single_cycle(g):
 seen=set();u=0
 for _ in range(len(g)):
  if u in seen:return False
  seen.add(u);u=int(g[u])
 return u==0 and len(seen)==len(g)
for d in sorted((R/'runs').glob('*')):
 gp=d/'maps/graphs.npz'
 if not gp.exists():continue
 graphs=np.load(gp)['test'];cycle=np.array([single_cycle(g) for g in graphs]);cm=np.repeat(cycle,10)
 for f in sorted((d/'maps').glob('*_fit*.npz')):
  if not f.with_suffix('.json').exists():continue
  z=np.load(f);hop={'stay':0,'one':1,'two':2}[f.stem.split('_')[0]];target=z['labels'][:,hop];pre=z['pre']==target;post=z['pred']==target;native=z['native']==target
  for name,mask in [('distinct',z['distinct']),('single_cycle',cm)]:
   assert name!='single_cycle' or np.all(z['distinct'][mask])
   n=int(mask.sum());both=int((pre&post&mask).sum());corrected=int((~pre&post&mask).sum());damaged=int((pre&~post&mask).sum());pc=int((post&mask).sum())
   out.append({'run':d.name,'map':f.stem,'population':name,'n':n,'n_cycle_graphs':int(cycle.sum()),'pre_pct':float(pre[mask].mean()*100),'post_pct':float(post[mask].mean()*100),'native_pct':float(native[mask].mean()*100),'both_correct':both,'corrected_by_F':corrected,'damaged_by_F':damaged,'fraction_final_correct_already_correct':both/pc if pc else None,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
s={'scope':'Structure-only restriction of the original locked test set to permutations consisting of one ten-node cycle. No training or model selection on this subset. Graphs and fits are not independent backbone replications.','rows':out}
(R/'analysis/cycle_audit.json').write_text(json.dumps(s,indent=2))
print(json.dumps([r for r in out if r['map'].startswith('two')],indent=2))
