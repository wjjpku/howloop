import sys,json,random,torch
from pathlib import Path
P=Path(__file__).resolve().parents[1];root=P.parent
used=set()
for folder in ['n10_migration_20260923','n10_fig4_fresh_20260924','n10_selected_mechanism_20260924','patch_search_20260926','mechanism_factorization_20260926','state_recombination_20260926']:
 f=root/folder/'datasets.json'
 for v in json.loads(f.read_text()).values():
  if isinstance(v,list) and v and isinstance(v[0],list):used.update(map(tuple,v))
for v in json.loads((root/'dense_affine_mechanism_20260925/graph_data.json').read_text()).values():
 if isinstance(v,list) and v and isinstance(v[0],list):used.update(map(tuple,v))
pool=sorted(set(map(tuple,torch.load(root/'n10_migration_20260923/locks/donors.pt',weights_only=False)['successors'].tolist()))-used);random.Random(2026092607).shuffle(pool)
assert len(pool)>322
(P/'datasets.json').write_text(json.dumps(dict(smoke=pool[:2],discovery=pool[2:66],confirmation=pool[66:322],seed=2026092607,excluded=len(used)),indent=2))
