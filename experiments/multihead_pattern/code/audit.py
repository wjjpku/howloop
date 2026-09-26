from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();data=json.loads((P/'datasets.json').read_text());sel=json.loads((P/'selection.json').read_text());new=set(map(tuple,data['discovery']+data['confirmation']));assert len(new)==320;checks={}
for old in ['patch_search_20260926','mechanism_factorization_20260926','state_recombination_20260926']:
 d=json.loads((P.parent/old/'datasets.json').read_text());prev=set()
 for v in d.values():
  if isinstance(v,list) and v and isinstance(v[0],list):prev.update(map(tuple,v))
 assert not prev&new
checks['prior_cohort_disjoint']=True
for n in ['B','D','E']:
 assert sel[n]['source_sha256']==sha(P/'discovery'/f'{n}_1.npz');orig=json.loads((P/'discovery'/f'{n}_1.json').read_text())
 for fit in [1,2]:
  m=json.loads((P/'confirmation'/f'{n}_{fit}.json').read_text());d=np.load(P/'confirmation'/f'{n}_{fit}.npz');assert m['weights_unchanged'] and m['same_run_exact'];assert m['hashes']['selection']==sha(P/'selection.json');assert all(orig['hashes'][k]==v for k,v in m['hashes'].items() if k!='selection');assert np.array_equal(d['graphs'],data['confirmation']);assert len(d['labels'])==2560;assert set(d['conditions'])==set(sel[n]['ids'])
  for scope in ['answer','all']:
   assert all(f'{scope}_0_{mask:03d}' in d['conditions'] for mask in range(256))
   ix=np.where(d['conditions']==f'{scope}_0_000')[0][0];assert np.array_equal(d['predictions'][ix],d['baseline'][0]);assert np.array_equal(d['wrong_current'][ix],d['baseline'][0])
 checks[n]=dict(fits=2,selection_hash_verified=True,baseline_exact=True,all_unattenuated_subsets_tested=True,weights_unchanged=True)
(P/'AUDIT.json').write_text(json.dumps(checks,indent=2));print(checks)
