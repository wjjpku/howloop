import json, itertools
from pathlib import Path
import torch
from reasoning_loop.paper2027_graph_g4_protocol import create_unique_lock, permutation_codes, load_unique_lock, LOCK_KIND, sha256
root=Path(__file__).resolve().parents[1]; out=root/'locks'; out.mkdir(exist_ok=True)
gen=torch.Generator().manual_seed(20260923)
used=set(); cohorts={}
def write(name,rows,role,seed):
 t=torch.tensor(rows,dtype=torch.long); codes=permutation_codes(t)
 assert len(set(codes.tolist()))==len(rows)
 p=out/(name+'.pt'); assert not p.exists(),p
 torch.save(dict(kind=LOCK_KIND,node_count=10,max_depth=8,permutations=len(rows),seed=seed,role=role,unique_permutations=True,universe_size=3628800,successors=t,permutation_codes=codes,starts_per_permutation=list(range(10))),p)
 load_unique_lock(p); return p
for name,count in [('selection',128),('discovery',32),('confirmation',512),('rings',512),('smoke',4)]:
 rows=[]
 while len(rows)<count:
  order=torch.randperm(10,generator=gen)
  if name=='rings':
   row=torch.empty(10,dtype=torch.long);row[order]=order.roll(-1)
  else:row=order
  key=tuple(row.tolist())
  if key in used:continue
  used.add(key);rows.append(list(key))
 cohorts[name]=rows
 write(name,rows,'selection' if name=='selection' else 'final_test',20260923)
# Exclude all single-edge-output-swap donor permutations before any training.
donors=set()
for name in ('discovery','confirmation','smoke'):
 for row in cohorts[name]:
  for i,j in itertools.combinations(range(10),2):
   g=row.copy();g[i],g[j]=g[j],g[i];key=tuple(g)
   if key not in used:donors.add(key)
write('donors',list(map(list,sorted(donors))),'final_test',20260923)
(root/'datasets.json').write_text(json.dumps(cohorts))
manifest={p.stem:{'sha256':sha256(p),'graphs':load_unique_lock(p)['permutations']} for p in sorted(out.glob('*.pt'))}
(root/'locks_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
