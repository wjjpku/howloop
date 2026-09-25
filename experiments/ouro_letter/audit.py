import json,hashlib,collections
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=[('semantic_confirmation64','confirm_semantics.py','shared_path_confirmation_with_cf.json','SEMANTIC_CONFIRMATION_PROTOCOL.md',13),('mediation_confirmation64','confirm_mediation.py','confirmation_pairs.json','CONFIRMATION_PROTOCOL.md',13),('restore_confirmation64','query_restore.py','restore_confirmation_with_cf.json','RESTORE_PROTOCOL.md',20)]
report={}
for name,code,data,protocol,narms in spec:
 m=json.loads((P/name/'manifest.json').read_text()); assert m['status']=='complete' and m['parameters_unchanged']
 for key,file in [('code_sha',code),('pairs_sha',data),('protocol_sha',protocol)]: assert m[key]==sha(P/file),(name,key)
 rows=[json.loads(x) for x in (P/name/'results.jsonl').read_text().splitlines()]; groups=collections.defaultdict(list)
 for r in rows:groups[r['condition']].append(r)
 assert len(groups)==narms and all(sorted(r['pair'] for r in g)==list(range(64)) for g in groups.values())
 gen=[r for r in rows if 'first_generated_token' in r]; assert all(r['first_generated_token']==r['prediction'] for r in gen)
 assert not any(r.get('truncated',False) for r in gen)
 checks={k:max(r.get('max_logit_error',0) for r in g) for k,g in groups.items() if 'self' in k or k=='restore_all'}
 assert all(v==0 for k,v in checks.items() if 'output' in k or k=='restore_all')
 report[name]={'rows':len(rows),'arms':len(groups),'pairs':64,'hashes_verified':True,'generated_runs':len(gen),'self_logit_errors':checks,'base_sha':m['base_sha'],'j_sha':m['j_sha']}
assert len({x['base_sha'] for x in report.values()})==len({x['j_sha'] for x in report.values()})==1
seen=set(); datasets={}
for file in ['discovery_pairs.json','confirmation_pairs.json','shared_path_discovery.json','shared_path_confirmation_with_cf.json','restore_confirmation_with_cf.json']:
 pairs=json.loads((P/file).read_text())['pairs']; own=set()
 for d in pairs:
  for side in ['base','source']:
   g=d[side+'_graph']; identity=tuple(sorted(g.items())); assert identity not in seen and identity not in own,(file,identity); own.add(identity)
   x=d[side+'_start']; visited=[]
   for _ in range(10):visited.append(x); x=g[x]
   assert len(set(visited))==10 and x==visited[0]
   assert visited[8]==d[side+'_answer']
  assert len(set(d['target_ids']))==3
  assert len(d['base']['ids'])==len(d['source']['ids'])
  slots=set(i for slot in d['base']['slots']+d['source']['slots'] for i in slot)
  assert all(a==b or i in slots for i,(a,b) in enumerate(zip(d['base']['ids'],d['source']['ids'])))
  if 'shared_prefix_hops' in d:
   b=s=d['source_start']
   for _ in range(7):b=d['base_graph'][b];s=d['source_graph'][s];assert b==s
   assert d['base_graph'][b]==d['counterfactual'] and d['source_graph'][s]==d['source_answer']
  assert len({d['base_answer'],d['source_answer'],d['counterfactual']})==3
 seen|=own;datasets[file]={'pairs':len(pairs),'unique_graphs':len(own)}
report['dataset_audit']=datasets;report['total_disjoint_graphs']=len(seen)
report['scope']='One frozen task-adapted backbone and one affine map; no training-set disjointness claim.'
(P/'AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
