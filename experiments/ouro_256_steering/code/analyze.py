import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];rows=[]
for shard in range(2):
 man=json.loads((P/f'shard{shard}/manifest.json').read_text());assert man['status']=='complete' and man['parameters_unchanged']
 rows.extend(json.loads(s) for s in (P/f'shard{shard}/results.jsonl').read_text().splitlines())
conditions=sorted(set(r['condition'] for r in rows));expected=set(range(256));data={}
for k in conditions:
 rr=sorted([r for r in rows if r['condition']==k],key=lambda r:r['pair']);assert len(rr)==256 and {r['pair'] for r in rr}==expected;data[k]=rr
rng=np.random.default_rng(2026092634);draw=rng.integers(0,256,(10000,256))
def wilson(k,n):
 z=1.959963984540054;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return [center-half,center+half]
def est(v):return dict(effect=float(v.mean()),ci=np.quantile(v[draw].mean(1),[.025,.975]).tolist())
summary={};aa={}
for k,rr in data.items():
 aa[k]=np.array([r['correct'] for r in rr],dtype=float);n=int(aa[k].sum());summary[k]=dict(correct=n,n=256,accuracy=n/256,wilson95=wilson(n,256))
 if 'generation_correct' in rr[0]:
  gen=np.array([r['generation_correct'] for r in rr]);summary[k].update(full_name_correct=int(gen.sum()),full_name_disagreements=int(np.count_nonzero(gen!=aa[k])),truncations=sum(r['truncated'] for r in rr))
 if 'self' in k:summary[k]['max_logit_error']=max(r['max_logit_error'] for r in rr)
contrasts={}
for x,y in [('selected_rescue_pattern','native'),('J','selected_damage_pattern'),('selected_rescue_pattern','selected_rescue_value'),('selected_rescue_pattern','selected_unrelated_pattern'),('selected_rescue_pattern','selected_wrong_call_pattern'),('selected_rescue_pattern','neighbor_rescue_pattern'),('J','selected_rescue_pattern')]:
 d=aa[x]-aa[y];contrasts[x+'__minus__'+y]={**est(d),'x_only':int((d==1).sum()),'y_only':int((d==-1).sum())}
den=(aa['J']-aa['native'])[draw].mean(1);num=(aa['selected_rescue_pattern']-aa['native'])[draw].mean(1);ratio=num[den>0]/den[den>0]
out=dict(conditions=summary,contrasts=contrasts,recovered_gain=dict(estimate=float((aa['selected_rescue_pattern']-aa['native']).mean()/(aa['J']-aa['native']).mean()),ci=np.quantile(ratio,[.025,.975]).tolist()))
(P/'summary.json').write_text(json.dumps(out,indent=2));(P/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda x:(x['pair'],x['condition']))));print(json.dumps(out,indent=2))
