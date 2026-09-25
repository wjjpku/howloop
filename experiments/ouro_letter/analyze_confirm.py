import argparse,json,collections
from pathlib import Path
import numpy as np
from uncertainty import wilson,paired_exact
p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();D=Path(a.out);m=json.loads((D/'manifest.json').read_text());assert m['status']=='complete'
rows=[json.loads(s) for s in (D/'results.jsonl').read_text().splitlines()];groups=collections.defaultdict(dict)
for x in rows:
 assert x['pair'] not in groups[x['condition']]
 groups[x['condition']][x['pair']]=x
keys=sorted(groups['native']);n=len(keys);assert all(sorted(v)==keys for v in groups.values());assert len(groups)==13
rng=np.random.default_rng(2026092309);ix=rng.integers(0,n,size=(10000,n));summary={};arrays={}
for k,g in groups.items():
 v=np.array([g[i]['correct'] for i in keys],float);arrays[k]=v;bs=v[ix].mean(1)
 s=dict(n=n,correct=int(v.sum()),accuracy=float(v.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),mean_probability=float(np.mean([g[i]['probability'] for i in keys])))
 if 'generation_correct' in g[keys[0]]:
  s.update(generation_correct=sum(g[i]['generation_correct'] for i in keys),truncated=sum(g[i]['truncated'] for i in keys));assert all(g[i]['first_generated_token']==g[i]['prediction'] for i in keys)
 if 'self' in k:s['max_logit_error']=max(g[i]['max_logit_error'] for i in keys)
 s['wilson_ci95']=wilson(int(v.sum()),n)
 summary[k]=s
pairs=[('late_rescue_pattern','native'),('J','late_damage_pattern')]+[('late_rescue_pattern',k) for k in ['late_rescue_value','late_unrelated_pattern','late_wrong_call_pattern','early_rescue_pattern','middle_rescue_pattern','local_rescue_pattern']]
contrasts={}
for a,b in pairs:
 d=arrays[a]-arrays[b];v=d[ix].mean(1);contrasts[a+'_minus_'+b]=dict(delta=float(d.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),a_only=int((d>0).sum()),b_only=int((d<0).sum()))
for c in contrasts.values():c.update(paired_exact(c['a_only'],c['b_only'],n))
den=float((arrays['J']-arrays['native']).mean());recovery=float((arrays['late_rescue_pattern']-arrays['native']).mean()/den) if den>0 else None
result=dict(conditions=summary,contrasts=contrasts,recovered_gain=recovery,bootstrap_replicates=10000,unit='paired base graph/task instance; one checkpoint and map')
(D/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
