from pathlib import Path
import numpy as np,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
P=Path(__file__).resolve().parents[1];data={};rows=[]
for n in ['B','D','E','C','A']:
 d=np.load(P/'raw'/f'{n}.npz');paths=d['paths'][:,:10];assert np.all(np.sort(paths,axis=1)==np.arange(10));pred=d['predictions'];depth=(pred[...,None]==paths).argmax(-1);freq=np.stack([(depth==k).mean(1) for k in range(10)]);prob=np.take_along_axis(d['probabilities'],paths[None],axis=2).mean(1).T;data[n]=dict(freq=freq,prob=prob,boundaries=d['boundaries'].tolist(),seed=json.loads((P/'raw'/f'{n}.json').read_text())['seed'])
 for i,b in enumerate(d['boundaries']):
  for k in range(10):rows.append([data[n]['seed'],str(b),k,float(freq[k,i]),float(prob[k,i])])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def figure(names,stages,filename,prob=False):
 if stages=='layers':idx=np.arange(2,41,2);ticks=[f'{i//2+1}\nL{i%2+1}' for i in range(20)];title='Native backbone: readout after each layer';xlabel='Loop / layer (after attention and MLP residual additions)';cut=11.5
 else:idx=np.arange(13,33);ticks=[f'{(i-1)//4+1}\nL{((i-1)%4)//2+1}'+(' A' if i%2 else ' M') for i in idx];title='Native backbone: within-layer readouts, loops 4–8';xlabel='A: after attention + residual     M: after MLP + residual';cut=11.5
 fig,axs=plt.subplots(len(names),1,figsize=(12.8,len(names)*2.65+.7),layout='constrained',squeeze=False)
 for ax,n in zip(axs[:,0],names):
  v=data[n]['prob' if prob else 'freq'][:,idx]*100;im=ax.imshow(v,origin='lower',vmin=0,vmax=100,cmap='Blues',aspect='auto',interpolation='nearest');ax.set_yticks(range(10));ax.set_ylabel(f"Seed {data[n]['seed']}\nPath depth k");ax.set_xticks(range(len(idx)),ticks,fontsize=8);ax.axvline(cut,color='#bd623b',ls='--',lw=1.5);ax.set_ylim(-.5,9.5)
  for j in range(len(idx)):
   for k in range(10):
    if v[k,j]>=10:ax.text(j,k,f'{v[k,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if v[k,j]>55 else '#17354D')
  ax.set_xticks(np.arange(-.5,len(idx),1),minor=True);ax.tick_params(which='minor',length=0);ax.grid(which='minor',axis='x',color='white',alpha=.25,lw=.5)
 axs[-1,0].set_xlabel(xlabel);fig.suptitle(title+'\nFrozen final readout head; dashed line = end of supervised loop 6',fontsize=13);fig.colorbar(im,ax=axs[:,0],shrink=.85,pad=.012,label='Mean probability (%)' if prob else 'Examples predicting f^k(s) (%)')
 fig.savefig(P/(filename+'.png'),dpi=180);fig.savefig(P/(filename+'.pdf'));plt.close(fig)
figure(['B','D','E'],'layers','native_layer_seed357');figure(['C','A'],'layers','native_layer_seed46');figure(['B','D','E'],'stages','native_sublayer_seed357');figure(['C','A'],'stages','native_sublayer_seed46');figure(['B','D','E'],'layers','native_layer_probability_seed357',True)
with (P/'readout_statistics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['backbone_seed','boundary','path_depth_mod10','prediction_fraction','mean_probability']);w.writerows(rows)
summary={n:{'seed':v['seed'],'boundaries':v['boundaries'],'modal_depth':v['freq'].argmax(0).tolist(),'modal_fraction':v['freq'].max(0).tolist(),'depth8':v['freq'][8].tolist(),'depth9':v['freq'][9].tolist()} for n,v in data.items()};(P/'summary.json').write_text(json.dumps(summary,indent=2))
for n,v in summary.items():
 print('SEED',v['seed'])
 for i,b in enumerate(v['boundaries']):
  if i>=13:print(b,v['modal_depth'][i],round(v['modal_fraction'][i]*100,1))
