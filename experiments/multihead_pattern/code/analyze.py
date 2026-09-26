import json,numpy as np,csv,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];selection=json.loads((P/'selection.json').read_text());out={};rows=[];rng=np.random.default_rng(92607);bs=rng.integers(0,256,size=(2000,256));fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
def heads(mask):return ', '.join(f'L{i//4+1}.H{i%4}' for i in range(8) if mask&(1<<i)) or 'none'
for col,(n,seed) in enumerate([('B',3),('D',5),('E',7)]):
 ds=[np.load(P/'confirmation'/f'{n}_{f}.npz') for f in [1,2]];y=ds[0]['labels'];assert np.array_equal(y,ds[1]['labels']);mask=(y[:,0]!=y[:,1])&(y[:,0]!=y[:,2])&(y[:,1]!=y[:,2]);den=mask.reshape(256,10).sum(1);metrics={}
 for i,k in enumerate(ds[0]['conditions']):
  assert k==ds[1]['conditions'][i];pred=np.stack([d['predictions'][i] for d in ds]);wrong=np.stack([d['wrong_current'][i] for d in ds]);good=(pred==y[:,1])&mask;num=good.reshape(2,256,10).sum(-1).mean(0);boot=num[bs].sum(1)/den[bs].sum(1);v=dict(mean=float(good.sum()/2/mask.sum()),fits=[float(x.sum()/mask.sum()) for x in good],ci95=np.quantile(boot,[.025,.975]).tolist(),wrong_current=float((wrong[:,mask]==y[mask,1]).mean()),heads=heads(int(k.split('_')[-1])));metrics[str(k)]=v
  rows.append([seed,k,v['heads'],int(mask.sum()),*v['fits'],v['mean'],*v['ci95'],v['wrong_current']])
 bas={key:float(np.mean([(d['baseline'][i,mask]==y[mask,1]).mean() for d in ds])) for i,key in enumerate(['native','full_J'])};out[n]=dict(seed=seed,n=int(mask.sum()),baseline=bas,metrics=metrics,selected=selection[n]['best_by_size'])
 for family,color,ls in [('all_0','#3D73A2','-'),('answer_0','#3D73A2','--'),('all_1','#C87946','-'),('answer_1','#C87946','--')]:
  vals=[metrics[f"{family}_{selection[n]['best_by_size'][family][str(k)]:03d}"]['mean']*100 for k in range(1,9)];axs[col].plot(range(1,9),vals,color=color,ls=ls,marker='o',markersize=3,label=family)
 axs[col].set_title(f'Seed {seed}');axs[col].set_xlabel('Number of patched heads');axs[col].set_ylim(-2,102);axs[col].set_xticks(range(1,9));axs[col].grid(alpha=.15);axs[col].spines[['top','right']].set_visible(False)
axs[0].set_ylabel('One-hop accuracy (%)');axs[2].legend(['All positions; native residual','Answer only; native residual','All positions; attenuated residual','Answer only; attenuated residual'],fontsize=7,loc='best');fig.savefig(P/'head_count_accuracy.png',dpi=190)
(P/'summary.json').write_text(json.dumps(out,indent=2))
with open(P/'all_confirmed_results.csv','w') as f:
 w=csv.writer(f);w.writerow(['seed','condition','heads','n','fit1','fit2','mean','ci_low','ci_high','wrong_current']);w.writerows(rows)
for n,v in out.items():
 print(n,'n=',v['n'],'baseline',v['baseline'])
 for fam in v['selected']:
  print(fam,[(k,heads(ms),round(v['metrics'][f'{fam}_{ms:03d}']['mean']*100,2)) for k,ms in v['selected'][fam].items()])
