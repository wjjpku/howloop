from pathlib import Path
import json,numpy as np
O=Path(__file__).parent;summary={};rng=np.random.default_rng(2026092502);boot=rng.integers(0,512,(5000,512))
for model in ['A','C','B','D','E']:
 ds=[np.load(O/f'{model}_fit{f}.npz') for f in [1,2]];l=ds[0]['labels'];mask=np.all(np.diff(np.sort(l,axis=1),axis=1)!=0,axis=1);den=mask.reshape(512,10).sum(1);rows={}
 for a,b in [('one','one'),('one','two'),('two','one'),('two','two')]:
  i=list(ds[0]['sequences']).index(a+'_'+b);hop={'one':1,'two':2};target=hop[a]+hop[b]
  hits=np.array([(d['second'][:,i]==l[:,target])&mask for d in ds]);g=hits.reshape(2,512,10).sum(2).mean(0);draw=g[boot].sum(1)/den[boot].sum(1)
  pred=[d['second'][:,i] for d in ds];first_idx=list(ds[0]['names']).index(a);valid=np.array([(d['first'][:,first_idx]==l[:,hop[a]])&mask for d in ds]);conditional=[float((hits[f]&valid[f]).sum()/valid[f].sum()) for f in range(2)]
  control_idx=list(ds[0]['sequences']).index(a+'_raw')
  rows[a+'_'+b]={'target_hop':target,'fit_accuracy':[float(x.sum()/mask.sum()) for x in hits],'mean':float(hits.sum()/(2*mask.sum())),'ci95':np.quantile(draw,[.025,.975]).tolist(),'first_correct_conditional':conditional,'first_accuracy':[float(x.sum()/mask.sum()) for x in valid],'no_second_J_same_target':float(np.mean([(d['second'][mask,control_idx]==l[mask,target]).mean() for d in ds])),'answer_distribution_hops0to4':[float(np.mean([(p[mask]==l[mask,h]).mean() for p in pred])) for h in range(5)]}
 summary[model]={'n_distinct_0to4':int(mask.sum()),'all_n':len(mask),'sequences':rows}
 print(model,'n',mask.sum())
 for k,v in rows.items(): print(k,'acc',round(v['mean']*100,2),'first',np.round(np.array(v['first_accuracy'])*100,2),'conditional',np.round(np.array(v['first_correct_conditional'])*100,2),'noJ',round(v['no_second_J_same_target']*100,2),'dist',np.round(np.array(v['answer_distribution_hops0to4'])*100,1))
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
