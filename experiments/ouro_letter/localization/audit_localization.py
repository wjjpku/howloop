import json,hashlib,collections
from pathlib import Path
p=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
configs={'localize_smoke':'smoke.json','localize_layers8':'layers.json','localize_combos8':'layer_combos.json','localize_headgroups8':'headgroups.json','localize_ranked8':'ranked_heads.json','localize_compact8':'compact.json','localize_ten_refine8':'ten_refine.json','localize_confirmation64':'confirmation.json'};out={}
for folder,config in configs.items():
 if not (p/folder/'manifest.json').exists():continue
 m=json.loads((p/folder/'manifest.json').read_text());assert m['status']=='complete',folder
 assert m['code_sha']==sha(p.parent/'localize_patterns.py') and m['config_sha']==sha(p/config)
 assert m['protocol_sha']==sha(p/'PROTOCOL.md') and m['parameters_unchanged']
 data=p/'localize_holdout64.json' if folder=='localize_confirmation64' else p.parent/'discovery_pairs.json';assert m['pairs_sha']==sha(data)
 n=64 if folder=='localize_confirmation64' else 1 if folder=='localize_smoke' else 8
 rows=[json.loads(x) for x in (p/folder/'results.jsonl').read_text().splitlines()];g=collections.defaultdict(dict)
 for r in rows:
  assert r['pair'] not in g[r['condition']];g[r['condition']][r['pair']]=r
  if 'first_generated_token' in r:assert r['first_generated_token']==r['prediction'] and not r['truncated']
 assert len(g)==len(m['conditions'])+2 and all(sorted(a)==list(range(n)) for a in g.values())
 for name,a in g.items():
  if name.endswith('self_output'):assert max(r['max_logit_error'] for r in a.values())==0
  if 'self' in name:assert all(r['prediction']==g['native'][i]['prediction'] for i,r in a.items())
 out[folder]={'pairs':n,'conditions':len(g),'rows':len(rows),'hashes_match':True,'generated':sum('first_generated_token' in r for r in rows)}
old=json.loads((p/'exclude_all.json').read_text())['pairs'];new=json.loads((p/'localize_holdout64.json').read_text())['pairs'];sig=lambda g:tuple(sorted(g.items()));oldset={sig(d[k]) for d in old for k in ['base_graph','source_graph']};fresh=[sig(d[k]) for d in new for k in ['base_graph','source_graph']];assert len(set(fresh))==128 and not oldset.intersection(fresh)
for d in new:
 for side in ['base','source']:
  g=d[side+'_graph'];x=d[side+'_start'];walk=[]
  for _ in range(10):walk.append(x);x=g[x]
  assert len(set(walk))==10 and x==walk[0] and walk[8]==d[side+'_answer']
 assert d['base']['slots']==d['source']['slots'];slots={i for s in d['base']['slots'] for i in s};assert len(d['base']['ids'])==len(d['source']['ids'])
 assert all(a==b or i in slots for i,(a,b) in enumerate(zip(d['base']['ids'],d['source']['ids'])))
out['holdout']={'previous_graphs':len(oldset),'new_graphs':128,'disjoint':True,'single_cycles':True,'token_slots_aligned':True}
(p/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
