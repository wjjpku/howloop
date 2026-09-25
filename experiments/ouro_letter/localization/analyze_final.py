import json,sys,collections,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from uncertainty import wilson,paired_exact
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete' and m['parameters_unchanged'];g=collections.defaultdict(dict)
for r in map(json.loads,(p/'results.jsonl').read_text().splitlines()):
 assert r['pair'] not in g[r['condition']];g[r['condition']][r['pair']]=r
assert len(g)==len(m['conditions'])+2 and all(sorted(a)==list(range(64)) for a in g.values())
s={};arrays={}
for name,a in g.items():
 vals=np.array([a[i]['correct'] for i in range(64)],int);arrays[name]=vals
 d={'correct':int(vals.sum()),'n':64,'accuracy':float(vals.mean()),'wilson95':wilson(int(vals.sum()),64),'mean_probability':float(np.mean([x['probability'] for x in a.values()]))}
 if any('first_generated_token' in x for x in a.values()):
  assert all(x['first_generated_token']==x['prediction'] for x in a.values());d['generation_correct']=sum(x['generation_correct'] for x in a.values());d['truncations']=sum(x['truncated'] for x in a.values());assert d['truncations']==0
 if 'self' in name:
  d['max_logit_error']=max(x['max_logit_error'] for x in a.values())
  assert np.array_equal(vals,arrays['native'])
  if name.endswith('output'):assert d['max_logit_error']==0
 s[name]=d
rng=np.random.default_rng(2026092919);ix=rng.integers(0,64,(10000,64));contrasts={}
for a,b in [('size16_rescue_pattern','native'),('J','size16_damage_pattern'),('size16_rescue_pattern','full_rescue_pattern'),('size16_rescue_pattern','selected_rescue_pattern'),('selected_rescue_pattern','native'),('J','selected_damage_pattern'),('selected_rescue_pattern','full_rescue_pattern'),('selected_damage_pattern','full_damage_pattern'),('selected_rescue_pattern','neighbor_rescue_pattern'),('selected_rescue_pattern','selected_unrelated_pattern'),('selected_rescue_pattern','selected_wrong_call_pattern')]:
 if a not in arrays or b not in arrays:continue
 d=arrays[a]-arrays[b];q={'delta':float(d.mean()),'bootstrap95':np.quantile(d[ix].mean(1),[.025,.975]).tolist()};q.update(paired_exact(int((d>0).sum()),int((d<0).sum()),64));contrasts[a+' minus '+b]=q
out={'conditions':s,'contrasts':contrasts,'n':64,'scope':'one backbone/map; new graph identities'};(p/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
