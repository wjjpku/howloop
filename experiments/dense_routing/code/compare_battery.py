from pathlib import Path
import json,gzip,numpy as np
from collections import defaultdict
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923');out={};rng=np.random.default_rng(260926);w=rng.multinomial(512,np.full(512,1/512),5000)
def load(p):
 groups=defaultdict(lambda:defaultdict(list))
 with gzip.open(p,'rt') as f:
  for line in f:
   d=json.loads(line)
   if d['receiver']!='J_one':continue
   for k,v in d.items():
    if isinstance(v,list):groups[(d['controller_seed'],d['kind'],d['condition'])][k].extend(v)
 return {k:{n:np.array(v) for n,v in row.items()} for k,row in groups.items()}
def stat(v,m,g):
 den=np.bincount(g,weights=m,minlength=512);num=np.bincount(g,weights=v*m,minlength=512);dd=w@den;bb=w@num;valid=dd>0
 return {'n':int(den.sum()),'mean':float(num.sum()/den.sum()) if den.sum() else None,'ci95':np.quantile(bb[valid]/dd[valid],[.025,.975]).tolist() if valid.any() else None}
for name in ['A','C','B','D','E']:
 head=json.loads((B/f'local/{name}/head_selection.json').read_text())['locked_head'];dp=P/f'battery/{name}/head{head}/confirmation/events.jsonl.gz';rp=P/f'reference/{name}/confirmation/events.jsonl.gz'
 if not dp.exists() or not rp.exists() or not dp.with_name('manifest.json').exists() or not rp.with_name('manifest.json').exists():continue
 ds,rs=load(dp),load(rp);rows={}
 specs=[('restore_selected','rescue',f'clean_context_H{head}','target'),('restore_control','rescue',f'clean_context_H{(head+1)%4}','target'),('restore_all','rescue','clean_context_H0123','target'),('pattern_rerouted','cross_graph_address',f'J_one_pat_H{head}','target'),('output_source','cross_graph_address',f'J_one_ctx_H{head}','donor_target'),('J_to_raw','unconditional_swap','rescue_H0123','one_target'),('raw_to_J','unconditional_swap','damage_H0123','one_target')]
 for label,kind,cond,target in specs:
  rows[label]={}
  for fit in [1,2]:
   d,r=ds[fit,kind,cond],rs[fit,kind,cond];assert np.array_equal(d['graph_ids'],r['graph_ids']);g=d['graph_ids']
   if kind=='rescue':md=d['eligible']&d['broken'];mr=r['eligible']&r['broken']
   elif kind=='cross_graph_address':md=d['eligible'];mr=r['eligible']
   else:
    bd=ds[fit,'baseline','clean'];br=rs[fit,'baseline','clean'];md=(bd['current']!=bd['one_target'])&(bd['current']!=bd['two_target'])&(bd['one_target']!=bd['two_target']);mr=md
   td=d[target] if target in d else ds[fit,'baseline','clean'][target];tr=r[target] if target in r else rs[fit,'baseline','clean'][target];a=d['prediction']==td;b=r['prediction']==tr;common=md&mr
   rows[label][str(fit)]={'dense_own':stat(a,md,g),'lowrank_own':stat(b,mr,g),'dense_common':stat(a,common,g),'lowrank_common':stat(b,common,g),'paired_delta_common':stat(a.astype(float)-b,common,g)}
 out[name]={'locked_head':head,'metrics':rows}
(P/'battery_comparison.json').write_text(json.dumps(out,indent=2));print('Compared battery:',list(out))
