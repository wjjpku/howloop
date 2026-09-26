from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];O=R/'outputs/audit';O.mkdir(parents=True,exist_ok=True)
old=json.loads((R/'experiments/target_exchange/values.json').read_text());vals={r['condition']:r for r in old if r['model']=='graph'}
s6=json.loads((R/'experiments/target_exchange/graph_summary.json').read_text())['A']
rows={6:{'bc':{c:vals[c]['mean']/100 for c in range(11)},'d':{},'counts':{'cross':[3966,3963],'steering':4044,'exchange':4116,'restore':[4413,4417]},'head':2}}
for recv,direction in [('one','two_to_one'),('two','one_to_two')]:
 for layer in ['own','L1','L2','L12']:
  x=s6['baseline'][recv] if layer=='own' else s6['interventions'][f'{direction}/{layer}/pattern'];rows[6]['d'][f'{direction}/{layer}']={k:x[k]['mean'] for k in ['one','two']}
for seed in [10,13]:
 d=json.loads((R/f'experiments/selected_mechanism/runs/seed{seed}/mechanism_summary.json').read_text());head=d['head_selection']['locked_head'];rr=[r for r in d['battery'] if r['selected']];bc={}
 def select(kind,condition,receiver='J_one'):return [r for r in rr if r['kind']==kind and r['condition']==condition and r['receiver']==receiver]
 def mean(rs,key='accuracy'):assert len(rs)==2;return float(np.mean([r[key] for r in rs]))
 for c,kind,cond,rec,key in [(3,'cross_graph_address',f'J_one_pat_H{head}','J_one','accuracy'),(4,'cross_graph_address',f'J_one_ctx_H{head}','J_one','accuracy'),(5,'cross_graph_address',f'J_one_pat_H{head}','J_one','donor_fraction'),(6,'cross_graph_address',f'J_one_ctx_H{head}','J_one','donor_fraction'),(7,'baseline','clean','identity','accuracy'),(8,'baseline','clean','J_one','accuracy'),(9,'unconditional_swap','rescue_H0123','J_one','accuracy'),(10,'unconditional_swap','damage_H0123','J_one','accuracy'),(0,'rescue',f'clean_context_H{head}','J_one','accuracy'),(1,'rescue',f'clean_context_H{(head+1)%4}','J_one','accuracy'),(2,'rescue','clean_context_H0123','J_one','accuracy')]:bc[c]=mean(select(kind,cond,rec),key)
 rows[seed]={'bc':bc,'d':{},'head':head,'counts':{'cross':[r['n'] for r in select('cross_graph_address',f'J_one_pat_H{head}')],'steering':4111,'exchange':4149,'restore':[r['n'] for r in select('rescue',f'clean_context_H{head}')]}}
 for recv,direction in [('one','two_to_one'),('two','one_to_two')]:
  for layer in ['own','L1','L2','L12']:
   cond=f'native_{recv}' if layer=='own' else f'{direction}/{layer}/pattern';xs=[r for r in d['target_exchange'] if r['condition']==cond];assert len(xs)==2;rows[seed]['d'][f'{direction}/{layer}']={k:float(np.mean([x[k] for x in xs])) for k in ['one','two']}
agg={'bc':{c:float(np.mean([r['bc'][c] for r in rows.values()])) for c in range(11)},'d':{c:{k:{'mean':float(np.mean(v:=[r['d'][c][k] for r in rows.values()])),'range':[min(v),max(v)],'seed_values':v} for k in ['one','two']} for c in rows[6]['d']}}
reference=json.loads((R/'experiments/revision/mechanism_aggregate.json').read_text())
for key in ['bc','d']:
 assert json.loads(json.dumps(agg[key]))==reference['aggregate'][key]
(O/'mechanism_aggregate.json').write_text(json.dumps({'seeds':rows,'aggregate':agg},indent=2)+'\n')
print('Three-backbone aggregate matches saved results.')
