from pathlib import Path
from collections import defaultdict
import sys,json,gzip,csv,numpy as np
P=Path(__file__).resolve().parents[1];seed=int(sys.argv[1]);out=P/'runs'/f'seed{seed}';selection=json.loads((out/'head_selection.json').read_text());rows=[]
for head in range(4):
 d=out/f'head{head}/confirmation/confirmation';groups=defaultdict(list);bases={}
 with gzip.open(d/'events.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);key=(r['kind'],r['receiver'],r['controller_seed'],r['condition']);groups[key].append(r)
   if r['kind']=='baseline' and r['receiver']=='J_one':
    for i,g in enumerate(r['graph_ids']):bases[(r['controller_seed'],g,i%10)]=(r['current'][i],r['one_target'][i],r['two_target'][i])
 for (kind,receiver,fit,condition),rs in groups.items():
  if kind not in ['baseline','unconditional_swap','cross_graph_address','block_error','rescue']:continue
  ns=np.zeros(512);ks=np.zeros(512);donor=orig=0
  for r in rs:
   for i,g in enumerate(r['graph_ids']):
    ok=r['eligible'][i]
    if kind=='rescue':ok=ok and r['broken'][i]
    if kind in ['baseline','unconditional_swap']:
     labels=bases[(fit,g,i%10)];ok=ok and len(set(labels))==3
    if ok:
     ns[g]+=1;ks[g]+=r['prediction'][i]==r['target'][i];orig+=r['prediction'][i]==r['original'][i]
     if 'donor_target' in r:donor+=r['prediction'][i]==r['donor_target'][i]
  n=int(ns.sum());k=int(ks.sum());ci=[None,None]
  if n:
   rng=np.random.default_rng(20260926);ix=rng.integers(0,512,size=(5000,512));den=ns[ix].sum(1);vals=ks[ix].sum(1)[den>0]/den[den>0];ci=np.quantile(vals,[.025,.975]).tolist()
  rows.append(dict(seed=seed,head=head,selected=head==selection['locked_head'],kind=kind,receiver=receiver,fit=fit,condition=condition,n=n,correct=k,accuracy=k/n if n else None,ci_low=ci[0],ci_high=ci[1],donor_fraction=donor/n if n else None,original_fraction=orig/n if n else None))
with (out/'mechanism_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
exchange=[]
for fit in [1,2]:
 z=np.load(P/f'exchange/seed{seed}_fit{fit}.npz');ys=z['labels'];mask=(ys[:,0]!=ys[:,1])&(ys[:,0]!=ys[:,2])&(ys[:,1]!=ys[:,2])
 for name,pred in list(zip(['native_raw','native_one','native_two'],z['native'].T))+list(zip(z['conditions'],z['predictions'])):
  counts=[int((pred[mask]==ys[mask,i]).sum()) for i in range(3)];exchange.append(dict(fit=fit,condition=str(name),n=int(mask.sum()),current=counts[0]/mask.sum(),one=counts[1]/mask.sum(),two=counts[2]/mask.sum()))
(out/'mechanism_summary.json').write_text(json.dumps(dict(seed=seed,head_selection=selection,battery=rows,target_exchange=exchange),indent=2))
print(json.dumps(dict(event='summary_complete',seed=seed,conditions=len(rows))),flush=True)
