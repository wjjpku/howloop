from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent
r=P/'matched'
s=json.loads((P/'matched/summary.json').read_text());assert s['complete'] and s['completed_fits']==60
rows=[];reference=None
for seed in range(5):
 a=json.loads((r/f'runs/final_seed{seed}/summary.json').read_text());b=json.loads((r/f'runs/stepwise_seed{seed}/summary.json').read_text());v=json.loads((r/f'runs/pair{seed}_validated.json').read_text())
 assert v['matched'] and a['initial_state_sha256']==b['initial_state_sha256']==v['initial_state_sha256']
 assert a['training_tokens_sha256']==b['training_tokens_sha256']==v['training_tokens_sha256']
 for regime in ['final','stepwise']:
  d=r/f'runs/{regime}_seed{seed}/maps';g=np.load(d/'graphs.npz')
  code=lambda ar:set(map(tuple,ar.tolist()))
  assert not(code(g['train']) & (code(g['selection'])|code(g['test']))) and not(code(g['selection']) & code(g['test']))
  if reference is None:reference={k:g[k].copy() for k in g.files}
  else:assert all(np.array_equal(g[k],reference[k]) for k in reference)
  def cycle(row):
   u=0;seen=set()
   for _ in range(10):
    if u in seen:return False
    seen.add(u);u=int(row[u])
   return u==0
  cm=np.repeat([cycle(x) for x in g['test']],10)
  for target,hop in [('stay',0),('one',1),('two',2)]:
   for fit in [1,2]:
    f=d/f'{target}_fit{fit}.npz';z=np.load(f);meta=json.loads(f.with_suffix('.json').read_text());m=z['distinct'];y=z['labels'][:,hop];assert m.sum()==4137 and len(y)==5120
    acc=(z['pred'][m]==y[m]).mean();assert abs(acc-meta['test_accuracy_distinct'])<1e-6
    rr=next(x for x in s['rows'] if (x['seed'],x['regime'],x['target'],x['fit'])==(seed,regime,target,fit))
    assert hashlib.sha256(f.read_bytes()).hexdigest()==rr['source_sha256'] and abs(acc-rr['full_J'])<1e-12
    rows.append({'seed':seed,'regime':regime,'target':target,'fit':fit,'native':float((z['native'][m]==y[m]).mean()),'accuracy':float(acc),'ring_accuracy':float((z['pred'][cm]==y[cm]).mean()),'n_ring':int(cm.sum())})
means={}
for regime in ['final','stepwise']:
 means[regime]={}
 for target in ['stay','one','two']:
  a=np.array([np.mean([x['accuracy'] for x in rows if x['seed']==seed and x['regime']==regime and x['target']==target]) for seed in range(5)])
  means[regime][target]={'mean_pct':float(a.mean()*100),'sd_pct':float(a.std(ddof=1)*100),'seed_pct':(a*100).tolist()}
report={'passed':True,'completed_fits':60,'checks':['paired initialization hashes','paired full token streams','all graph pools identical across models','train selection test graph disjointness','raw prediction accuracies match summaries','all raw prediction hashes match'],'means':means,'rows':rows}
reference=json.loads((P/'matched/paper_audit.json').read_text())
assert reference['means']==means
out=P.parents[1]/'outputs/audit';out.mkdir(parents=True,exist_ok=True)
(out/'matched_supervision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(means,indent=2))
