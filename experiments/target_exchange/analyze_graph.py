from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];out=P/'analysis';out.mkdir(exist_ok=True);summary={};rows=[];rng=np.random.default_rng(2026092505)
for backbone in ['A','C','B','D','E']:
 files=[P/'graph'/f'{backbone}_fit{fit}.npz' for fit in [1,2]]
 if not all(p.exists() for p in files):continue
 ds=[np.load(f) for f in files];lab=ds[0]['labels'];mask=(lab[:,0]!=lab[:,1])&(lab[:,0]!=lab[:,2])&(lab[:,1]!=lab[:,2]);ng=len(lab)//10
 weights=rng.multinomial(ng,np.full(ng,1/ng),size=5000).astype(float);counts=mask.reshape(ng,10).sum(1);denom=weights@counts
 def metric(predictions,target):
  hit=np.mean([p==target for p in predictions],axis=0)*mask
  gs=hit.reshape(ng,10).sum(1);draws=weights@gs/denom
  return {'mean':float(hit.sum()/mask.sum()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'fit_values':[float((p[mask]==target[mask]).mean()) for p in predictions]}
 stats={}
 for j,name in enumerate(ds[0]['conditions']):
  preds=[d['predictions'][j] for d in ds];vals={label:metric(preds,lab[:,i]) for i,label in enumerate(['current','one','two'])};vals['other_mean']=1-sum(vals[k]['mean'] for k in ['current','one','two']);stats[str(name)]=vals
  for target in ['current','one','two']:rows.append({'backbone':backbone,'condition':str(name),'target':target,'n':int(mask.sum()),**{k:vals[target]['mean' if k=='accuracy' else 'ci95'][0 if k=='ci_low' else 1] if k!='accuracy' else vals[target]['mean'] for k in ['accuracy','ci_low','ci_high']}})
 baseline={}
 for j,name in enumerate(['raw','one','two']):baseline[name]={labname:metric([d['native'][:,j] for d in ds],lab[:,i]) for i,labname in enumerate(['current','one','two'])}
 summary[backbone]={'n_distinct':int(mask.sum()),'graphs':ng,'baseline':baseline,'interventions':stats}
(out/'graph_summary.json').write_text(json.dumps(summary,indent=2))
with (out/'graph_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,(src,dst) in zip(axs,[('two','one'),('one','two')]):
 comps=['pattern','value','pattern_value','query','key','query_key','mlp','block_input'];sites=['L1','L2','L12'];z=np.array([[summary['A']['interventions'][f'{src}_to_{dst}/{l}/{c}'][src]['mean'] for l in sites] for c in comps]);im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(3),sites);ax.set_yticks(range(len(comps)),comps);ax.set_title(f'{src}-hop patterns/state into {dst}-hop run')
 for i in range(len(comps)):
  for j in range(3):ax.text(j,i,f'{z[i,j]*100:.1f}',ha='center',va='center',color='white' if z[i,j]>.65 else 'black',fontsize=9)
fig.colorbar(im,ax=axs,label='Donor-target accuracy');fig.savefig(out/'graph_target_switch.pdf');fig.savefig(out/'graph_target_switch.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);names=list(summary);conditions=['raw','J','L1 pattern','L2 pattern','L12 pattern','L12 pattern+value'];width=.12
for i,cond in enumerate(conditions):
 vals=[]
 for b in names:
  s=summary[b];v=s['baseline']['raw' if cond=='raw' else 'one']['one']['mean'] if cond in ['raw','J'] else s['interventions']['one_to_raw/'+{'L1 pattern':'L1/pattern','L2 pattern':'L2/pattern','L12 pattern':'L12/pattern','L12 pattern+value':'L12/pattern_value'}[cond]]['one']['mean'];vals.append(v*100)
 ax.bar(np.arange(len(names))+(i-2.5)*width,vals,width,label=cond)
ax.set_xticks(range(len(names)),[f'{b} (seed '+str({'A':6,'B':3,'C':4,'D':5,'E':7}[b])+')' for b in names]);ax.set_ylabel('One-hop accuracy (%)');ax.set_ylim(0,105);ax.legend(ncol=3,fontsize=9);fig.savefig(out/'graph_backbone_pathways.pdf');fig.savefig(out/'graph_backbone_pathways.png',dpi=180)
print(json.dumps({'backbones':list(summary),'rows':len(rows)}))
