import json,collections,sys
from pathlib import Path
import numpy as np
from uncertainty import wilson,paired_exact
D=Path(sys.argv[1]);assert json.loads((D/'manifest.json').read_text())['status']=='complete'
g=collections.defaultdict(dict)
for line in (D/'results.jsonl').read_text().splitlines():
 r=json.loads(line);assert r['pair'] not in g[r['condition']];g[r['condition']][r['pair']]=r
keys=sorted(g['base']);assert len(g)==9 and all(sorted(x)==keys for x in g.values());n=len(keys)
a={k:np.array([v[i]['correct'] for i in keys],bool) for k,v in g.items()};broken=a['base']&~a['corrupt'];bn=int(broken.sum());rng=np.random.default_rng(2026092327);ix=rng.integers(0,bn,size=(10000,bn)) if bn else None
out={}
for k,v in g.items():
 x=a[k];y=x[broken];d={'n':n,'correct':int(x.sum()),'broken_n':bn,'recovered':int(y.sum()),'conditional_recovery':float(y.mean()) if bn else None,'conditional_ci95':np.quantile(y[ix].mean(1),[.025,.975]).tolist() if bn else None}
 if 'generation_correct' in v[keys[0]]:d.update(generation_correct=sum(v[i]['generation_correct'] for i in keys),truncations=sum(v[i]['truncated'] for i in keys),first_token_name_disagreements=sum(v[i]['generation_correct']!=v[i]['correct'] for i in keys))
 d['conditional_wilson_ci95']=wilson(int(y.sum()),bn)
 out[k]=d
if bn:
 h5=a['restore_selected'][broken].astype(float);other=np.array([a['restore_neighbor'][broken]],float);diff=h5[ix].mean(1)-other[:,ix].mean(2).max(0)
 compare={'selected_minus_neighbor':float(h5.mean()-other.mean(1).max()),'ci95_reselect_best_other_each_bootstrap':np.quantile(diff,[.025,.975]).tolist()}
else:compare={'not_identifiable':'query corruption broke no initially correct cases'}
r={'conditions':out,'comparison':compare,'all_head_max_logit_error':max(v['max_logit_error'] for v in g['restore_all'].values()),'bootstrap_replicates':10000};(D/'analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
