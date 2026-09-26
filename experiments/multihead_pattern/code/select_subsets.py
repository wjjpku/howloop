import numpy as np,json,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1];result={}
for n in ['B','D','E']:
 d=np.load(P/'discovery'/f'{n}_1.npz');y=d['labels'];eligible=(y[:,0]!=y[:,1])&(y[:,0]!=y[:,2])&(y[:,1]!=y[:,2]);acc=(d['predictions'][:,eligible]==y[eligible,1]).mean(1);scores=dict(zip(d['conditions'],acc));selected=set();chosen={}
 for scope in ['answer','all']:
  for scale in [0,1]:
   family=f'{scope}_{scale}';chosen[family]={}
   for count in range(1,9):
    candidates=[k for k in range(1,256) if k.bit_count()==count];best=max(candidates,key=lambda k:(scores[f'{family}_{k:03d}'],-k));chosen[family][str(count)]=best;selected.add(f'{family}_{best:03d}')
    for h in range(8):
     if best&(1<<h):selected.add(f'{family}_{best^(1<<h):03d}')
   for group,candidates in [('L1',[k for k in range(1,16)]),('L2',[k<<4 for k in range(1,16)]),('cross_pair',[ (1<<i)|(1<<j) for i in range(4) for j in range(4,8)])]:
    best=max(candidates,key=lambda k:(scores[f'{family}_{k:03d}'],-k));chosen[family][group]=best;selected.add(f'{family}_{best:03d}')
   for k in [0,15,240,255]+[1<<h for h in range(8)]:selected.add(f'{family}_{k:03d}')
 result[n]=dict(ids=sorted(selected),best_by_size=chosen,n_eligible=int(eligible.sum()),source_sha256=hashlib.sha256((P/'discovery'/f'{n}_1.npz').read_bytes()).hexdigest())
assert not (P/'selection.json').exists();(P/'selection.json').write_text(json.dumps(result,indent=2));print({n:len(v['ids']) for n,v in result.items()})
