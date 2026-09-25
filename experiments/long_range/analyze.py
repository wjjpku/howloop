from pathlib import Path
import json,hashlib,csv
import numpy as np
P=Path(__file__).resolve().parent;D=P/'results';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gm=json.loads((D/'graph/manifest.json').read_text());assert gm['status']=='complete' and gm['graphs']==8192 and len(gm['models'])==12
counts=np.stack([np.load(D/f'graph/seed{s}.npz')['correct_counts'] for s in range(100,112)]);assert counts.shape==(12,3,40,8192) and counts.min()>=0 and counts.max()<=8
assert sha(D/'graph/successors.npy')==gm['graph_sha256']
# Compare every overlap graph with the archived evaluator, without selecting new results.
oldroot=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/sec09_coverage/horizon_decomposition_20260921/raw')
old=json.loads((D/'old_lock_successors.json').read_text());lookup={tuple(r):i for i,r in enumerate(old)};graphs=np.load(D/'graph/successors.npy');common=[(i,lookup[tuple(row)]) for i,row in enumerate(graphs) if tuple(row) in lookup];audit=[]
for si,seed in enumerate(range(100,112)):
 for mi,sub in enumerate(['raw','rank48_seed1/full','rank48_seed2/full']):
  with (oldroot/f'seed{seed}'/sub/'permutation_clusters.csv').open() as stream:
   val={(int(r['permutation']),int(r['call'])):int(r['moving_successor_correct']) for r in csv.DictReader(stream) if int(r['call'])<=40}
  different=sum(int(counts[si,mi,t-1,ni])!=val[oi,t] for ni,oi in common for t in range(1,41))
  audit.append({'seed':seed,'condition':sub,'comparisons':len(common)*40,'different_counts':int(different)})
(D/'all_old_new_overlap_audit.json').write_text(json.dumps(audit,indent=2))
raw=counts[:,0].mean(axis=2)/8;full=counts[:,1:].mean(axis=(1,3))/8
rng=np.random.default_rng(2026092403);choice=rng.integers(0,12,(10000,12));lo,hi=np.quantile(full[choice].mean(1),[.025,.975],axis=0)
summary={'graph':{'n_graphs':8192,'n_starts_per_graph':8,'n_backbones':12,'n_J_per_backbone':2,'raw_by_seed':{},'with_J_by_seed':{},'means':{}}}
for band,(a,b) in {'9_16':(8,16),'17_32':(16,32),'17_40':(16,40)}.items():
 for mode,v in [('raw',raw),('with_J',full)]:
  values=v[:,a:b].mean(1);summary['graph'][mode+'_by_seed'][band]=values.tolist();summary['graph']['means'][mode+'_'+band]=float(values.mean())
summary['graph']['all_backbones_decline']=bool(np.all(full[:,8:16].mean(1)>full[:,16:32].mean(1)))
summary['graph']['all_above_raw_17_32']=bool(np.all(full[:,16:32].mean(1)>raw[:,16:32].mean(1)))
np.savez(D/'graph_curve_summary.npz',raw=raw.mean(0),full=full.mean(0),low=lo,high=hi,raw_by_seed=raw,full_by_seed=full)
rows_by_key={};duplicates=0
for directory in ['parity0','parity1','parity2','helper0','helper1']:
 if not (D/directory/'manifest.json').exists():continue
 m=json.loads((D/directory/'manifest.json').read_text());assert m['status'] in ['complete','partial_superseded']
 rr=[json.loads(x) for x in (D/directory/'results.jsonl').read_text().splitlines()] if (D/directory/'results.jsonl').exists() else []
 if m['status']=='complete':assert len(rr)==2*len(m['lengths'])
 else:rr=[r for r in rr if (D/directory/f"length_{r['length']}.npz").exists()]
 for r in rr:
  assert r['examples']==128 and r['loop']==r['length'] and r['exact_match']==r['correct']/128
  cc=np.load(D/f"{directory}/length_{r['length']}.npz")['correct'];assert cc.shape==(2,128);assert cc[['raw','J'].index(r['variant'])].sum()==r['correct']
  key=(r['variant'],r['length'])
  if key in rows_by_key:
   assert rows_by_key[key]['correct']==r['correct'];duplicates+=1
  else:rows_by_key[key]=r
rows=list(rows_by_key.values())
assert len(rows)==202 and len({(r['variant'],r['length']) for r in rows})==202
assert sorted({r['length'] for r in rows})==list(range(500,1001,5))
rows=sorted(rows,key=lambda r:(r['variant'],r['length']))
with (D/'parity_dense_extension.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary['parity']={'lengths':101,'examples_per_length':128,'step':5,'total_paired_inputs':12928,'at_1000':{r['variant']:r['exact_match'] for r in rows if r['length']==1000}}
(D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
