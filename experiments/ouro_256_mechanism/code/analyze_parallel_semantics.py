import json,collections,argparse
from pathlib import Path
import numpy as np
from uncertainty import wilson,paired_exact
p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();D=Path(a.out);manifest=json.loads((D/'manifest.json').read_text());assert manifest['status']=='complete'
rows=[json.loads(s) for s in (D/'results.jsonl').read_text().splitlines()];g=collections.defaultdict(dict)
for r in rows:assert r['pair'] not in g[r['condition']];g[r['condition']][r['pair']]=r
keys=sorted(g['base']);n=len(keys);assert len(g)==11 and all(sorted(x)==keys for x in g.values())
rng=np.random.default_rng(2026092317);ix=rng.integers(0,n,size=(10000,n));summary={};arrays={}
for kind,data in g.items():
 arr=np.array([[data[i]['prediction']==data[i]['target_ids'][j] for j in range(3)] for i in keys],int);assert (arr.sum(1)<=1).all();arrays[kind]=arr
 d={'n':n,'base':int(arr[:,0].sum()),'source':int(arr[:,1].sum()),'counterfactual':int(arr[:,2].sum()),'other':int(n-arr.sum()),'ci95':np.quantile(arr[ix].mean(1),[.025,.975],axis=0).tolist()}
 if 'generation_matches' in data[keys[0]]:
  gen=np.array([data[i]['generation_matches'] for i in keys],int);d['generation_counts']=gen.sum(0).tolist();d['generation_other']=int(n-gen.sum());d['truncations']=sum(data[i]['truncated'] for i in keys);d['first_token_name_disagreements']=int((gen!=arr).any(1).sum())
 if 'self' in kind:d['max_logit_error']=max(data[i]['max_logit_error'] for i in keys)
 d['wilson_ci95']=[wilson(int(arr[:,j].sum()),n) for j in range(3)]
 summary[kind]=d
contrasts={}
for name,aa,bb,col in [('pattern_minus_output_counterfactual','selected_source_pattern','selected_source_output',2),('output_minus_pattern_source','selected_source_output','selected_source_pattern',1),('pattern_minus_wrong_call_counterfactual','selected_source_pattern','selected_wrong_call_pattern',2),('pattern_minus_neighbor_base_route','selected_source_pattern','neighbor_source_pattern',2)]:
 d=arrays[aa][:,col]-arrays[bb][:,col];contrasts[name]={'delta':float(d.mean()),'ci95':np.quantile(d[ix].mean(1),[.025,.975]).tolist(),'a_only':int((d>0).sum()),'b_only':int((d<0).sum())}
for c in contrasts.values():c.update(paired_exact(c['a_only'],c['b_only'],n))
result={'conditions':summary,'paired_contrasts':contrasts,'bootstrap_replicates':10000,'unit':'paired graph/input; one backbone and map'};(D/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
